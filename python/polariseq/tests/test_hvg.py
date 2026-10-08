"""Highly-variable-gene selection against scanpy's `flavor="seurat"`.

The rank-position diagnostic on heart_10k (chapter 09) traced a Jaccard
0.984 gap with scanpy to one cause: our mean-bin edges. pandas' `pd.cut`
builds `linspace(min, max, n+1)` and nudges only the *first* edge down by
0.1 % of the range; we used to widen every bin instead, which placed ~0.7 %
of genes one bin higher than scanpy and perturbed each bin's z-scores
enough to reorder the HVG tail. With pandas' edge rule the two agree to
|Δ dispersions_norm| < 5e-7 and the masks are identical on heart_10k and
combat_10k.

Two guards so this cannot regress:
- a CI-portable check against a numpy port of scanpy's exact algorithm
  (no scanpy/pandas dependency), and
- a real-data check against actual scanpy when both are available.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import polariseq as ps
import pytest
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
N_TOP, N_BINS = 500, 20


def _synthetic_lognormal(n_cells: int = 400, n_genes: int = 3000, seed: int = 0) -> sp.csr_matrix:
    """Counts shaped like scRNA-seq, put through the canonical
    normalize_total(1e4) + log1p, as float32 CSR."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(2.0, 0.3, size=(n_cells, n_genes)) + 0.02
    counts = rng.poisson(lam).astype(np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    scale = np.where(totals > 0, 1e4 / np.maximum(totals, 1.0), 1.0)
    return sp.csr_matrix(np.log1p(counts * scale))


def _scanpy_port_hvg(x: sp.csr_matrix, n_top: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """numpy port of scanpy's seurat-flavour HVG (its f64 stats path), with
    pandas' `pd.cut` edge rule inlined. Returns (mask, dispersions_norm)."""
    n = x.shape[0]
    data = x.tocsc()
    mean = np.zeros(x.shape[1])
    var = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        v = np.expm1(data.data[data.indptr[j] : data.indptr[j + 1]].astype(np.float64))
        mean[j] = v.sum() / n
        var[j] = (np.dot(v, v) / n - mean[j] ** 2) * n / (n - 1)
    mean[mean == 0] = 1e-12
    disp = np.log(var / mean, where=var / mean > 0, out=np.full(x.shape[1], np.nan))
    mean_log = np.log1p(mean)

    edges = np.linspace(mean_log.min(), mean_log.max(), n_bins + 1)
    edges[0] -= (mean_log.max() - mean_log.min()) * 0.001
    bins = np.clip(np.searchsorted(edges, mean_log, side="left") - 1, 0, n_bins - 1)

    dnorm = np.full(x.shape[1], np.nan)
    for k in np.unique(bins):
        m = bins == k
        d = disp[m]
        ok = np.isfinite(d)
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            dnorm[np.flatnonzero(m)[ok]] = 1.0
            continue
        z = (d[ok] - d[ok].mean()) / d[ok].std(ddof=1)
        dnorm[np.flatnonzero(m)[ok]] = z
    cut = np.sort(np.nan_to_num(dnorm, nan=-np.inf))[::-1][n_top - 1]
    return np.nan_to_num(dnorm, nan=-np.inf) >= cut, dnorm


class TestHvgMatchesScanpy:
    def test_mask_and_norm_match_scanpy_port(self) -> None:
        x = _synthetic_lognormal()
        r = ps.SparseMatrix.from_scipy(x).highly_variable_genes(N_TOP, N_BINS, True)
        mask = np.asarray(r["mask"], bool)
        dnorm = np.asarray(r["dispersions_norm"], np.float64)
        port_mask, port_dnorm = _scanpy_port_hvg(x, N_TOP, N_BINS)

        assert mask.sum() == N_TOP
        np.testing.assert_array_equal(
            mask,
            port_mask,
            err_msg="HVG mask disagrees with the scanpy port — bin edges drifted?",
        )
        fin = np.isfinite(dnorm) & np.isfinite(port_dnorm)
        np.testing.assert_allclose(dnorm[fin], port_dnorm[fin], atol=1e-5)

    @pytest.mark.skipif(
        find_spec("scanpy") is None or not (REPO / "data/bench/heart_10k.h5ad").exists(),
        reason="needs scanpy and the gitignored heart_10k dataset",
    )
    def test_achieved_level_against_scanpy_on_heart_10k(self) -> None:
        """The level achieved by the pandas-edge fix: identical masks on
        heart_10k (Jaccard 1.0, was 0.984). Guards the regression on real
        data where the boundary-adjacent genes actually occur."""
        import scanpy as sc

        adata = sc.read_h5ad(REPO / "data/bench/heart_10k.h5ad")
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        x = adata.X.copy().tocsr()
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        sc_mask = np.asarray(adata.var["highly_variable"], bool)

        r = ps.SparseMatrix.from_scipy(x).highly_variable_genes(2000, 20, True)
        ps_mask = np.asarray(r["mask"], bool)
        inter = (sc_mask & ps_mask).sum()
        union = (sc_mask | ps_mask).sum()
        assert inter / union >= 0.999, f"HVG Jaccard vs scanpy regressed: {inter / union:.4f}"


# ---------------------------------------------------------------- flavor="scarf"
FIXTURE_10K = REPO / "tests/data/medium_10kx2k.h5ad"


def _scarf_port(counts: np.ndarray, top_n: int, min_cells: int) -> np.ndarray:
    """Scarf's mark_hvgs (BSD-3-Clause, Dhapola and Karlsson) in numpy and statsmodels, as in
    scarf/feat_utils.py:fit_lowess, on counts scaled to 10,000 per cell."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x = counts / counts.sum(axis=1, keepdims=True) * 1e4
    mean, var, det = x.mean(0), x.var(0, ddof=1), (x > 0).sum(0)
    ok = (mean > 0) & (var > 0)
    la, lb = np.log(mean[ok]), np.log(var[ok])
    edges = np.histogram(la, bins=200)[1]
    edges[-1] += 0.1
    which = np.digitize(la, edges) - 1
    bins = [np.flatnonzero(which == i) for i in range(200)]
    bins = [b for b in bins if len(b)]
    floor = np.array([[lb[b][np.argmin(lb[b])], la[b][np.argmin(lb[b])]] for b in bins]).T
    fit = lowess(floor[0], floor[1], return_sorted=False, frac=0.1, it=100)
    c = np.zeros(ok.sum())
    for f, b in zip(fit, bins, strict=True):
        c[b] = np.exp(lb[b] - f)
    c_var = np.zeros(len(mean))
    c_var[ok] = c
    valid = np.flatnonzero((det > min_cells) & (c_var > 0))
    top = valid[np.lexsort((valid, -c_var[valid]))][:top_n]
    mask = np.zeros(len(mean), dtype=bool)
    mask[top] = True
    return mask


def _counts(n_cells: int = 600, n_genes: int = 1500, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam = rng.gamma(1.5, 0.4, size=(n_cells, n_genes)) * rng.gamma(2.0, 0.5, size=(1, n_genes))
    counts = rng.poisson(lam).astype(np.float32)
    counts[:, :5] = 0
    counts[:3, :5] = 400.0          # five genes seen in three cells only, with huge counts
    return counts


def _lognorm_adata(counts: np.ndarray):
    adata = ps.AnnData(X=sp.csr_matrix(counts))
    ps.pp.normalize_total(adata, target_sum=1e4)
    ps.pp.log1p(adata)
    return adata


class TestHvgScarf:
    def test_matches_a_port_of_scarfs_own_code(self) -> None:
        pytest.importorskip("statsmodels")
        counts = _counts()
        adata = _lognorm_adata(counts)
        ps.pp.highly_variable_genes(adata, n_top_genes=200, flavor="scarf", min_cells=6)
        ours = np.asarray(adata.var["highly_variable"]).astype(bool)
        ref = _scarf_port(counts, 200, 6)
        assert ours.sum() == 200
        assert (ours & ref).sum() / (ours | ref).sum() >= 0.99

    def test_rarely_detected_genes_are_left_out(self) -> None:
        counts = _counts()
        seurat = _lognorm_adata(counts)
        ps.pp.highly_variable_genes(seurat, n_top_genes=200, flavor="seurat")
        scarf = _lognorm_adata(counts)
        ps.pp.highly_variable_genes(scarf, n_top_genes=200, flavor="scarf")
        assert np.asarray(seurat.var["highly_variable"])[:5].any(), "seurat ranks the rare genes high"
        assert not np.asarray(scarf.var["highly_variable"])[:5].any()
        assert int(scarf.uns["hvg"]["min_cells"]) == 6   # 1% of 600 cells
        assert (np.asarray(scarf.var["detected"])[np.asarray(scarf.var["highly_variable"])] > 6).all()

    def test_out_of_core_selects_the_same_genes(self, proj) -> None:
        if not FIXTURE_10K.exists():
            pytest.skip("fixture not present")
        mem = ps.read_h5ad(FIXTURE_10K)
        ooc = ps.read_h5ad(FIXTURE_10K, mode="out-of-core")
        for a in (mem, ooc):
            ps.pp.normalize_total(a, target_sum=1e4)
            ps.pp.log1p(a)
            ps.pp.highly_variable_genes(a, n_top_genes=1000, flavor="scarf")
        np.testing.assert_array_equal(np.asarray(mem.var["highly_variable"]),
                                      np.asarray(ooc.var["highly_variable"]))


def test_seurat_with_a_floor_ranks_only_the_genes_above_it() -> None:
    counts = _counts()
    floored = _lognorm_adata(counts)
    ps.pp.highly_variable_genes(floored, n_top_genes=200, flavor="seurat", min_cells=6)
    mask = np.asarray(floored.var["highly_variable"]).astype(bool)
    assert not mask[:5].any(), "genes seen in three cells are below the floor"
    # the same as seurat on the genes above the floor alone
    detected = (counts > 0).sum(axis=0)
    keep = detected > 6
    sub = _lognorm_adata(counts)[:, np.flatnonzero(keep)]
    ps.pp.highly_variable_genes(sub, n_top_genes=200, flavor="seurat")
    np.testing.assert_array_equal(mask[keep], np.asarray(sub.var["highly_variable"]).astype(bool))
    # without a floor: unchanged (scanpy's behaviour)
    plain = _lognorm_adata(counts)
    ps.pp.highly_variable_genes(plain, n_top_genes=200, flavor="seurat")
    assert np.asarray(plain.var["highly_variable"])[:5].any()
