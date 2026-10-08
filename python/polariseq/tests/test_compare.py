"""Comparing conditions (``tl.compare_conditions``, ``get.pseudobulk``).

The pseudobulk test must be edgeR and limma's: ``tests/data/limma_reference.npz``
holds pseudobulk sums of Kang et al. (2018) for two cell types (one with
replicates missing a condition) and what edgeR 4.10 and limma 3.68 found on
them. The raw counts must give the same answer wherever they are read from:
the raw matrix, the on-disk store, or the source files once the matrix is
normalized in memory, and after cells are filtered.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
stats = pytest.importorskip("scipy.stats")

REFERENCE = "tests/data/limma_reference.npz"


def test_pseudobulk_test_is_edger_and_limma():
    f = np.load(REFERENCE)
    # Each row is already a replicate's sum: one "cell" per unit.
    a = ps.AnnData(X=sp.csr_matrix(f["counts"]),
                   obs={"cell_type": f["cell_type"], "condition": f["condition"],
                        "donor": f["donor"]},
                   var_names=list(f["genes"]))
    got = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type",
                                   sample_key="donor", min_cells=1)
    want = pd.DataFrame({"cell_type": f["ref_type"], "gene": f["ref_gene"],
                         "logFC": f["ref_logfc"], "t_ref": f["ref_t"], "p": f["ref_p"],
                         "adjp": f["ref_adjp"], "ave": f["ref_aveexpr"]})
    m = got.astype({"cell_type": str}).merge(want, on=["cell_type", "gene"], how="outer",
                                             indicator=True)
    assert (m["_merge"] == "both").all(), "the same genes pass edgeR's filter"
    np.testing.assert_allclose(m["log2_fold_change"], m["logFC"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m["t"], m["t_ref"], rtol=1e-9)
    np.testing.assert_allclose(m["p_value"], m["p"], rtol=1e-9)
    np.testing.assert_allclose(m["adj_p_value"], m["adjp"], rtol=1e-9)
    groups = a.uns["compare_conditions"]["groups"].set_index("group")
    assert groups["paired"].all()
    assert groups.loc["Megakaryocytes", "samples"] < groups.loc["CD14+ Monocytes", "samples"]


def _simulated(path, seed: int = 5) -> str:
    """Two cell types, six donors, each measured in both conditions; gene 0
    rises fourfold with treatment in type A only."""
    rng = np.random.default_rng(seed)
    rows, obs = [], []
    for donor in range(6):
        for cond in ("control", "treated"):
            for ct in ("A", "B"):
                n = 40 + 10 * donor
                lam = np.full(60, 1.5) * rng.lognormal(0, 0.2, 60)
                if ct == "A" and cond == "treated":
                    lam[0] *= 4
                rows.append(rng.poisson(lam, size=(n, 60)))
                obs += [(f"d{donor}", cond, ct)] * n
    x = np.vstack(rows).astype(np.float32)
    o = pd.DataFrame(obs, columns=["donor", "condition", "cell_type"])
    o.index = [f"c{i}" for i in range(len(o))]
    a = ad.AnnData(sp.csr_matrix(x), obs=o.astype("category"),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(60)]))
    a.write_h5ad(path)
    return str(path)


@pytest.fixture(scope="module")
def sim(tmp_path_factory) -> str:
    return _simulated(tmp_path_factory.mktemp("compare") / "sim.h5ad")


def _stats(t):
    return t.sort_values(["cell_type", "gene"])[
        ["log2_fold_change", "t", "p_value", "adj_p_value"]].to_numpy()


def test_the_raw_counts_are_found_wherever_they_are(sim, proj):
    args = dict(groupby="cell_type", sample_key="donor")
    raw = ps.read_h5ad(sim, mode="memory")
    want = ps.tl.compare_conditions(raw, "condition", "control", **args)
    top = want[want["cell_type"] == "A"].iloc[0]
    assert top["gene"] == "g0" and top["log2_fold_change"] > 1.5 and top["adj_p_value"] < 1e-3
    # Normalized in memory: the counts are read back from the file.
    norm = ps.read_h5ad(sim, mode="memory")
    ps.pp.normalize_total(norm)
    ps.pp.log1p(norm)
    np.testing.assert_array_equal(_stats(ps.tl.compare_conditions(norm, "condition", "control",
                                                                  **args)), _stats(want))
    # Out of core: the store's raw counts, under the normalization.
    ooc = ps.read_h5ad(sim, mode="out-of-core")
    ps.pp.normalize_total(ooc)
    np.testing.assert_array_equal(_stats(ps.tl.compare_conditions(ooc, "condition", "control",
                                                                  **args)), _stats(want))


def test_filtered_cells_are_the_cells_compared(sim):
    a = ps.read_h5ad(sim, mode="memory")
    drop = np.asarray(a.obs["donor"] == "d0")
    ps.pp.normalize_total(a)
    ps.pp._subset_cells(a, ~drop)  # as filter_cells removes cells
    got = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type",
                                   sample_key="donor")
    b = ps.read_h5ad(sim, mode="memory")
    b = b[np.asarray(b.obs["donor"] != "d0")]
    want = ps.tl.compare_conditions(b, "condition", "control", groupby="cell_type",
                                    sample_key="donor")
    np.testing.assert_array_equal(_stats(got), _stats(want))


def test_the_cell_level_test_is_welchs(sim):
    a = ps.read_h5ad(sim, mode="memory")
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    got = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type")
    # In float64: scipy computes float32 input in float32.
    x = a.X.to_scipy().toarray().astype(np.float64)
    ct, cond = np.asarray(a.obs["cell_type"]), np.asarray(a.obs["condition"])
    r = stats.ttest_ind(x[(ct == "B") & (cond == "treated")], x[(ct == "B") & (cond == "control")],
                        equal_var=False)
    g = got[got["cell_type"] == "B"].set_index("gene").loc[[f"g{j}" for j in range(60)]]
    np.testing.assert_allclose(g["t"], r.statistic, rtol=1e-9)
    np.testing.assert_allclose(g["p_value"], r.pvalue, rtol=1e-8)


def test_pseudobulk_sums_are_the_counts_summed(sim, proj):
    a = ps.read_h5ad(sim, mode="memory", name="sim")
    pb = ps.get.pseudobulk(a, "donor", groupby="cell_type", condition_key="condition")
    x = a.X.to_scipy().toarray()
    frame = pd.DataFrame(x).assign(ct=np.asarray(a.obs["cell_type"]),
                                   cond=np.asarray(a.obs["condition"]),
                                   donor=np.asarray(a.obs["donor"]))
    want = frame.groupby(["ct", "cond", "donor"], observed=True).sum()
    assert pb.shape == (24, 60)
    for i in range(pb.n_obs):
        key = (pb.obs["cell_type"][i], pb.obs["condition"][i], pb.obs["donor"][i])
        np.testing.assert_array_equal(np.asarray(pb.X)[i], want.loc[key].to_numpy())
    assert isinstance(pb.X, np.memmap)
    few = ps.get.pseudobulk(a, "donor", groupby="cell_type", condition_key="condition",
                            min_cells=60)
    assert few.n_obs == int((np.asarray(pb.obs["n_cells"]) >= 60).sum())


def test_ranking_against_a_reference_group_is_scanpys(sim):
    sc = pytest.importorskip("scanpy")
    a = ps.read_h5ad(sim, mode="memory")
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.tl.rank_genes_groups(a, "condition", reference="control", n_genes=60)
    b = a.to_anndata()
    sc.tl.rank_genes_groups(b, "condition", reference="control", method="t-test", n_genes=60)
    want = sc.get.rank_genes_groups_df(b, group="treated").set_index("names")
    ours = a.uns["rank_genes_groups"]["treated"]
    got = pd.DataFrame({k: ours[k] for k in ("scores", "logfoldchanges", "pvals")},
                       index=ours["names"]).loc[want.index]
    np.testing.assert_allclose(got["scores"], want["scores"], rtol=1e-5)
    np.testing.assert_allclose(got["logfoldchanges"], want["logfoldchanges"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(got["pvals"], want["pvals"], rtol=1e-4)
    assert list(a.uns["rank_genes_groups"]) == ["treated"]


def test_comparison_plots_draw(sim):
    import matplotlib

    matplotlib.use("Agg")
    a = ps.read_h5ad(sim, mode="memory")
    res = ps.tl.compare_conditions(a, "condition", "control", groupby="cell_type",
                                   sample_key="donor")
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.pca(a, n_components=5)
    ps.pp.neighbors(a, n_neighbors=10)
    ps.tl.umap(a)
    axes = ps.pl.violin(a, ["g0", "g1"], groupby="cell_type", split_by="condition", show=False)
    assert len(axes) == 2
    ps.pl.dotplot(a, ["g0", "g1", "g2"], groupby="cell_type", split_by="condition", show=False)
    ps.pl.volcano(res, "A", show=False)
    ps.pl.volcano(a, "A", show=False)  # from uns
    ps.pl.fold_changes(res, genes=["g0", "g1"], show=False)
    panels = ps.pl.umap(a, color="cell_type", split_by="condition", show=False)
    assert len([p for p in panels if p.get_visible()]) == 2
    with pytest.raises(ValueError, match="pick a group"):
        ps.pl.volcano(res, show=False)
