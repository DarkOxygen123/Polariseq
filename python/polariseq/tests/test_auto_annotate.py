"""Cell types named from marker genes: ``tl.auto_annotate`` and ``ps.marker_table``.

Synthetic counts with known types: the clusters get the type whose markers
they express, a cluster expressing none stays unassigned, a gene a type should
not express counts against it, two types that tie are named by their shared
Cell Ontology parent, and a small group of cells of another type inside a
cluster is reported. The scores equal a direct computation of the published
formulas, in memory and out of core. The shipped tables load and narrow by
tissue; mistakes are errors that say what is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polariseq as ps
import pytest
import scipy.sparse as sp
from polariseq import _annotate, _markers

N_BG = 150
A = [f"AG{i}" for i in range(10)]
B = [f"BG{i}" for i in range(10)]
C = [f"CG{i}" for i in range(10)]
NEG = [f"NG{i}" for i in range(5)]
GENES = [f"G{i}" for i in range(N_BG)] + A + B + C + NEG
MARKERS = {"Alpha cells": A, "Beta cells": B, "Gamma cells": C}


def counts(seed: int = 0, n: int = 120, rare: int = 15, neg_in_a: bool = False):
    """Three clusters of ``n`` cells: a expresses A, b expresses B (``rare`` of
    its cells also C), c expresses none of the markers."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(3 * n, len(GENES))).astype(np.float32)
    col = {g: i for i, g in enumerate(GENES)}
    X[:, [col[g] for g in A + B + C + NEG]] = rng.poisson(0.2, size=(3 * n, 35))
    X[:n, [col[g] for g in A]] += rng.poisson(8.0, size=(n, len(A)))
    X[n:2 * n, [col[g] for g in B]] += rng.poisson(8.0, size=(n, len(B)))
    X[n:n + rare, [col[g] for g in C]] += rng.poisson(12.0, size=(rare, len(C)))
    if neg_in_a:
        X[:n, [col[g] for g in NEG]] += rng.poisson(8.0, size=(n, len(NEG)))
    labels = np.repeat(["a", "b", "c"], n)
    return X, labels


def make(mode: str, tmp_path, **kw):
    X, labels = counts(**kw)
    a = ps.AnnData(X=ps.SparseMatrix.from_scipy(sp.csr_matrix(X)),
                   obs={"cl": pd.Categorical(labels)},
                   obs_names=[f"c{i}" for i in range(X.shape[0])], var_names=GENES)
    if mode == "memory":
        return a
    path = tmp_path / "synthetic.h5ad"
    a.write_h5ad(path)
    b = ps.read_h5ad(str(path), mode=mode)
    b.obs["cl"] = pd.Categorical(labels)
    return b


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_clusters_get_the_type_they_express(proj, tmp_path, mode) -> None:
    a = make(mode, tmp_path)
    table = ps.tl.auto_annotate(a, "cl", source=None, markers=MARKERS)
    got = dict(zip(table["cluster"], table["cell_type"], strict=True))
    assert got == {"a": "Alpha cells", "b": "Beta cells", "c": "unassigned"}
    assert list(a.obs["cell_type"][:3]) == ["Alpha cells"] * 3
    row = table.set_index("cluster").loc["a"]
    assert row["support"] > 0.9 and row["score"] > 0
    # the rare cells inside cluster b: named in its row, and by their own label
    assert "Gamma cells (15)" in table.set_index("cluster").loc["b", "also"]
    per_cell = np.asarray(a.obs["cell_type_cell"])
    assert (per_cell[120:135] == "Gamma cells").all()
    for k in ("cell_type_ontology_id", "cell_type_score", "cell_type_cell_score"):
        assert len(a.obs[k]) == a.n_obs
    assert set(a.uns["annotation"]) == {"table", "scores", "markers", "params"}


def test_scores_equal_the_published_formula(proj, tmp_path) -> None:
    """Cluster scores = mean over cells of sum_g z_g w_g / sqrt(sum w) (positive
    markers) minus the same over negative markers, z standardized over all cells."""
    a = make("memory", tmp_path)
    ps.tl.auto_annotate(a, "cl", source=None,
                        markers={"Alpha cells": {"positive": A, "negative": B[:3]}, "Beta cells": B})
    X, labels = counts()
    norm = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    Z = (norm - norm.mean(0)) / norm.std(0)
    used = a.uns["annotation"]["markers"]
    col = {g: i for i, g in enumerate(GENES)}
    for ct in ("Alpha cells", "Beta cells"):
        m = used[used["cell_type"] == ct]
        score = np.zeros(X.shape[0])
        for sign in (1, -1):
            part = m[m["sign"] == sign]
            if len(part):
                w = part["weight"].to_numpy()
                score += sign * Z[:, [col[g] for g in part["gene"]]] @ w / np.sqrt(w.sum())
        want = pd.Series(score).groupby(labels).mean()
        got = a.uns["annotation"]["scores"][ct]
        np.testing.assert_allclose(got.loc[["a", "b", "c"]].to_numpy(), want.to_numpy(),
                                   rtol=1e-4, atol=1e-4)


def test_memory_and_out_of_core_agree(proj, tmp_path) -> None:
    a, b = make("memory", tmp_path), make("out-of-core", tmp_path)
    ta = ps.tl.auto_annotate(a, "cl", source=None, markers=MARKERS)
    tb = ps.tl.auto_annotate(b, "cl", source=None, markers=MARKERS)
    pd.testing.assert_frame_equal(ta.drop(columns=["also"]), tb.drop(columns=["also"]),
                                  check_exact=False, rtol=1e-4)


def test_a_gene_the_type_should_not_express_counts_against_it(proj, tmp_path) -> None:
    a = make("memory", tmp_path, neg_in_a=True)
    markers = {"Plain": A[:5], "Excluding": {"positive": A[:5], "negative": NEG}}
    table = ps.tl.auto_annotate(a, "cl", source=None, markers=markers)
    assert table.set_index("cluster").loc["a", "cell_type"] == "Plain"
    scores = a.uns["annotation"]["scores"]
    assert scores.loc["a", "Excluding"] < scores.loc["a", "Plain"]


def test_two_types_that_tie_get_their_shared_parent(proj, tmp_path) -> None:
    a = make("memory", tmp_path)
    t = pd.DataFrame({"cell_type": ["CD4 T"] * 10 + ["CD8 T"] * 10, "gene": A + A,
                      "cl_id": ["CL:0000624"] * 10 + ["CL:0000625"] * 10})
    row = ps.tl.auto_annotate(a, "cl", source=None, markers=t).set_index("cluster").loc["a"]
    assert row["cell_type"] not in ("CD4 T", "CD8 T")
    shared = row["ontology_id"]
    assert shared in _markers.ancestors("CL:0000624") and shared in _markers.ancestors("CL:0000625")
    assert row["note"] == "between CD4 T and CD8 T" or row["note"] == "between CD8 T and CD4 T"
    # margin=0 keeps the best type itself
    row0 = ps.tl.auto_annotate(a, "cl", source=None, markers=t, margin=0).set_index("cluster").loc["a"]
    assert row0["cell_type"] in ("CD4 T", "CD8 T")


def test_accordion_weights() -> None:
    """ECs = log10(evidence) + 1; SPs rescaled from (n types .. 1) to the evidence
    range, log10 + 1; weight = their product; each type keeps its best ones."""
    t = pd.DataFrame({"type": ["x", "x", "y", "y"], "gene": ["g1", "g2", "g2", "g3"],
                      "col": [0, 1, 1, 2], "sign": 1, "evidence": [4, 1, 1, 2], "strength": 0.0})
    w = _annotate.weigh(t, None).set_index(["type", "gene"])
    # g1, g3: one type each (flipped to 2 -> evidence max 4); g2: both types (-> min 1)
    assert w.loc[("x", "g1"), "ecs"] == pytest.approx(np.log10(4) + 1)
    assert w.loc[("x", "g1"), "sps_reg"] == pytest.approx(np.log10(4) + 1)
    assert w.loc[("x", "g2"), "sps_reg"] == pytest.approx(1.0)
    assert w.loc[("x", "g1"), "weight"] == pytest.approx((np.log10(4) + 1) ** 2)
    top = _annotate.weigh(t, 1)
    assert sorted(zip(top["type"], top["gene"], strict=True)) == [("x", "g1"), ("y", "g3")]


def test_shipped_tables_load_and_narrow_by_tissue() -> None:
    for source in ("hpa", "msigdb_c8", "curated"):
        t = ps.marker_table(source)
        assert len(t) > 100 and set(_markers.COLUMNS) <= set(t.columns)
        assert not t["gene"].str.match(r"^(RP[LS]\d+[A-Z]?\d*$|MT-)").any()  # never markers
    hpa = ps.marker_table("hpa")
    assert {"Plasmacytoid dendritic cells", "Respiratory ionocytes", "Tuft cells"} <= set(hpa["cell_type"])
    lung = ps.marker_table("hpa", tissue="lung")
    assert "Respiratory ionocytes" in set(lung["cell_type"])
    assert "Hepatocytes" not in set(lung["cell_type"])
    assert "T cells" in set(lung["cell_type"])  # immune cells are in every tissue
    both = ps.marker_table(["hpa", "curated"])
    assert set(both["source"]) == {"hpa", "curated"}
    assert ps.marker_table("c8").equals(ps.marker_table("msigdb_c8"))
    assert _markers.match_cl("Plasmacytoid dendritic cells") == "CL:0000784"
    assert _markers.match_cl("Alpha cell") == "CL:0000171"  # pancreatic, not a retinal neuron


def test_your_own_markers_in_every_form(tmp_path) -> None:
    gmt = tmp_path / "m.gmt"
    gmt.write_text("Alpha cells\tna\tAG0\tAG1\nBeta cells\tna\tBG0\n", encoding="utf-8")
    csv = tmp_path / "m.csv"
    pd.DataFrame({"Cell_Type": ["Alpha cells", "Alpha cells"], "Gene": ["AG0", "NG0"],
                  "sign": ["+", "negative"]}).to_csv(csv, index=False)
    assert len(_markers.user_table(gmt)) == 3
    t = _markers.user_table(csv)
    assert list(t["sign"]) == [1, -1]
    t = _markers.user_table({"Alpha cells": {"positive": ["AG0"], "negative": ["NG0"]}})
    assert list(t["sign"]) == [1, -1]
    with pytest.raises(KeyError, match="positive"):
        _markers.user_table({"Alpha cells": {"pos": ["AG0"]}})
    with pytest.raises(KeyError, match="cell_type and gene"):
        _markers.user_table(pd.DataFrame({"type": ["x"], "symbol": ["AG0"]}))


def test_mistakes_say_what_is_wrong(proj, tmp_path) -> None:
    a = make("memory", tmp_path)
    with pytest.raises(KeyError, match="no marker source named 'hpaa'.*'hpa'"):
        ps.tl.auto_annotate(a, "cl", source="hpaa")
    with pytest.raises(KeyError, match="none of the .* marker genes"):
        ps.tl.auto_annotate(a, "cl", source=None, markers={"X": ["NOT_A_GENE"]})
    with pytest.raises(KeyError, match="no markers for tissue 'kneecap'"):
        ps.tl.auto_annotate(a, "cl", source="hpa", tissue="kneecap")
    with pytest.raises(KeyError, match="max_markers"):
        ps.tl.auto_annotate(a, "cl", source=None, markers=MARKERS, max_markers=0)
    with pytest.raises(KeyError, match="cluster|column"):
        ps.tl.auto_annotate(a, "no_such_column", source=None, markers=MARKERS)


def test_a_failed_download_says_where_to_put_the_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POLARISEQ_CACHE", str(tmp_path))

    def refuse(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(_markers.urllib.request, "urlopen", refuse)
    _markers._source_table.cache_clear()
    with pytest.raises(RuntimeError, match=r"save it as .*PanglaoDB_markers_27_Mar_2020\.tsv\.gz"):
        ps.marker_table("panglaodb")
    assert not list(tmp_path.rglob("*.part"))


def test_species_from_the_gene_symbols() -> None:
    assert _markers.guess_species(["Cd3e", "Ms4a1", "Lyz2", "Gapdh"]) == "mouse"
    assert _markers.guess_species(["CD3E", "MS4A1", "LYZ", "GAPDH"]) == "human"


def test_human_markers_map_to_mouse_genes(monkeypatch) -> None:
    monkeypatch.setattr(_markers, "_human_to_mouse", lambda: {"CD3E": ["Cd3e"], "LYZ": ["Lyz1", "Lyz2"]})
    t = pd.DataFrame({"cell_type": ["T", "Mono", "Mono"], "gene": ["CD3E", "LYZ", "CD14"], "sign": 1,
                      "evidence": 1, "strength": 0.0, "tissue": "any", "cl_id": "",
                      "species": "human", "source": "hpa"})
    m = _markers.to_species(t, "mouse")
    assert sorted(m["gene"]) == ["Cd14", "Cd3e", "Lyz1", "Lyz2"]  # no ortholog listed: by case


def test_the_names_are_saved_and_exported(proj, tmp_path) -> None:
    a = make("out-of-core", tmp_path)
    table = ps.tl.auto_annotate(a, "cl", source=None, markers=MARKERS)
    a.save("named")
    b = ps.load(proj.path, "named")
    assert list(b.obs["cell_type"]) == list(a.obs["cell_type"])
    pd.testing.assert_frame_equal(b.uns["annotation"]["table"], table)
    pd.testing.assert_frame_equal(b.uns["annotation"]["scores"], a.uns["annotation"]["scores"])
    out = a.write_h5ad(tmp_path / "named.h5ad")
    ad = pytest.importorskip("anndata")
    c = ad.read_h5ad(out)
    assert list(c.obs["cell_type"]) == list(a.obs["cell_type"])
    assert list(c.uns["annotation"]["table"]["cell_type"]) == list(table["cell_type"])


def test_tissue_words_match_whole_words_in_order() -> None:
    k = _markers.tissue_key
    assert k("outer medullary collecting duct") == "kidney"  # not "uterus"
    assert k("temporal lobe") == "brain"  # not "oral"
    assert k("choroid plexus") == "brain" and k("pigmented layer of retina") == "eye"
    assert k("cortex of thymus") == "immune" and k("adrenal cortex") == "endocrine"
    assert k("umbilical cord blood") == "bone_marrow" and k("GI tract") == "gut"
    assert k("Connective tissue") == "any" and k("stromal") == "any"
