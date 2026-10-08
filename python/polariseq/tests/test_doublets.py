"""Doublet detection (``ps.pp.scrublet``).

Cells of three types are drawn as Poisson counts, and doublets are made by
adding a cell of one type to a cell of another, as a droplet with two cells
does. The scores must rank those doublets above the singlets, be the same in
memory and out of core, score each batch on its own, and, with a project
folder, be memory-mapped files the Rust core wrote.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
metrics = pytest.importorskip("sklearn.metrics")

N_GENES = 400


def _cells(rng, n: int, kind: int) -> np.ndarray:
    lam = np.full(N_GENES, 0.2)
    lam[kind * 60:(kind + 1) * 60] = 6.0
    return rng.poisson(lam, size=(n, N_GENES)).astype(np.float32)


def _capture(rng, n: int, n_doublets: int) -> tuple[np.ndarray, np.ndarray]:
    kinds = rng.integers(0, 3, n)
    x = np.vstack([_cells(rng, 1, k) for k in kinds])
    a = rng.integers(0, 3, n_doublets)
    b = (a + rng.integers(1, 3, n_doublets)) % 3
    d = np.vstack([_cells(rng, 1, i) + _cells(rng, 1, j) for i, j in zip(a, b, strict=True)])
    return np.vstack([x, d]), np.r_[np.zeros(n, bool), np.ones(n_doublets, bool)]


@pytest.fixture(scope="module")
def path(tmp_path_factory) -> str:
    rng = np.random.default_rng(11)
    parts = [_capture(rng, 900, 60), _capture(rng, 700, 45)]
    x = np.vstack([p[0] for p in parts])
    truth = np.concatenate([p[1] for p in parts])
    lane = np.repeat(["lane1", "lane2"], [len(p[1]) for p in parts])
    obs = pd.DataFrame({"lane": pd.Categorical(lane), "truth": truth},
                       index=[f"c{i}" for i in range(len(truth))])
    p = tmp_path_factory.mktemp("doublets") / "cells.h5ad"
    ad.AnnData(X=sp.csr_matrix(x), obs=obs,
               var=pd.DataFrame(index=[f"g{j}" for j in range(N_GENES)])).write_h5ad(p)
    return str(p)


def test_doublets_rank_above_singlets(path):
    a = ps.read_h5ad(path, mode="memory")
    ps.pp.scrublet(a, batch_key="lane")
    truth = np.asarray(a.obs["truth"])
    assert metrics.roc_auc_score(truth, np.asarray(a.obs["doublet_score"])) > 0.95
    info = a.uns["scrublet"]
    assert info["batched_by"] == "lane" and set(info["batches"]) == {"lane1", "lane2"}
    for b in info["batches"].values():
        assert b["n_sim"] == 2 * b["n_cells"]
        assert 0 < b["threshold"] < 1
    called = np.asarray(a.obs["predicted_doublet"])
    assert called[truth].mean() > 0.5 and called[~truth].mean() < 0.05


def test_memory_and_out_of_core_agree_and_results_are_mapped(path, proj):
    mem = ps.read_h5ad(path, mode="memory", name="mem")
    ps.pp.scrublet(mem, batch_key="lane")
    ooc = ps.read_h5ad(path, mode="out-of-core", name="ooc")
    ps.pp.normalize_total(ooc)  # out of core the raw counts remain readable
    ps.pp.scrublet(ooc, batch_key="lane")
    np.testing.assert_array_equal(np.asarray(mem.obs["doublet_score"]),
                                  np.asarray(ooc.obs["doublet_score"]))
    score = ooc.obs["doublet_score"]
    assert isinstance(score, np.memmap)
    assert "runs/ooc/results/scrublet" in str(score.filename).replace("\\", "/")
    # Copy-on-write: changing the view leaves the file as the core wrote it.
    first = float(score[0])
    score[0] = 99.0
    assert float(np.load(score.filename, mmap_mode="r")[0]) == first


def test_each_batch_is_scored_alone(path):
    a = ps.read_h5ad(path, mode="memory")
    ps.pp.scrublet(a, batch_key="lane")
    lane2 = np.asarray(a.obs["lane"] == "lane2")
    b = a[lane2]
    ps.pp.scrublet(b, batch_key="lane")
    np.testing.assert_array_equal(np.asarray(a.obs["doublet_score"])[lane2],
                                  np.asarray(b.obs["doublet_score"]))


def test_normalized_counts_are_refused_in_memory(path):
    a = ps.read_h5ad(path, mode="memory")
    ps.pp.normalize_total(a)
    with pytest.raises(ValueError, match="raw counts"):
        ps.pp.scrublet(a)


def test_the_distribution_plot_draws(path):
    import matplotlib

    matplotlib.use("Agg")
    a = ps.read_h5ad(path, mode="memory")
    ps.pp.scrublet(a, batch_key="lane")
    axes = ps.pl.scrublet_score_distribution(a, show=False)
    assert axes.shape == (2, 2)
