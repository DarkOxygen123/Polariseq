"""The fetal-atlas preparation and the published-annotation comparison.

The campaign runs these on a 4-million-cell file for hours; a defect in the
comparison must not be found there. Everything here runs on a few hundred
synthetic cells, except the published marker table, which is the real file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import fetal_biology as fb  # noqa: E402
import fetal_prepare as fp  # noqa: E402


# ----------------------------------------------------------- preparation


def _blocks(rng, n_blocks=3, rows=40, genes=25):
    import scipy.sparse as sp

    return [sp.random(rows, genes, density=0.2, format="csr", dtype=np.float32,
                      random_state=rng + i) for i in range(n_blocks)]


def test_written_file_round_trips_through_both_readers(tmp_path: Path) -> None:
    import h5py
    import scipy.sparse as sp
    from anndata.io import sparse_dataset

    import anndata as ad

    blocks = _blocks(0)
    path = tmp_path / "fetal_test.h5ad"
    with h5py.File(path, "w") as f:
        fp._write_empty_anndata(f, blocks[0])
        X = sparse_dataset(f["X"])
        for b in blocks[1:]:
            X.append(b)
        fp._finish_anndata(
            f, [f"c{i}" for i in range(120)],
            pd.DataFrame({"gene_id": [f"ENSG{i}" for i in range(25)]},
                         index=pd.Index([f"g{i}" for i in range(25)], name="gene")),
            {"source": "test"})
    a = ad.read_h5ad(path)
    assert a.shape == (120, 25)
    assert abs(a.X - sp.vstack(blocks)).max() == 0
    assert json.loads(a.uns["provenance"])["source"] == "test"
    assert list(a.obs_names[:2]) == ["c0", "c1"]


def test_indptr_is_64_bit(tmp_path: Path) -> None:
    """The 4M file passes 2**31 non-zeros; anndata starts indptr at int32."""
    import h5py

    path = tmp_path / "x.h5ad"
    with h5py.File(path, "w") as f:
        fp._write_empty_anndata(f, _blocks(1)[0])
        assert f["X/indptr"].dtype == np.int64
        assert f["X/indices"].dtype == np.int32
        assert f["X/data"].compression == "gzip"


def test_subsets_are_nested_and_keep_their_cells(tmp_path: Path, monkeypatch) -> None:
    import h5py
    import scipy.sparse as sp
    from anndata.io import sparse_dataset

    import anndata as ad

    blocks = _blocks(2, n_blocks=5, rows=100, genes=30)
    full = sp.vstack(blocks).tocsr()
    monkeypatch.setattr(fp, "DATA", tmp_path)
    monkeypatch.setattr(fp, "FULL", tmp_path / "fetal500.h5ad")
    with h5py.File(fp.FULL, "w") as f:
        fp._write_empty_anndata(f, blocks[0])
        X = sparse_dataset(f["X"])
        for b in blocks[1:]:
            X.append(b)
        fp._finish_anndata(f, [f"c{i}" for i in range(500)],
                           pd.DataFrame(index=pd.Index([f"g{i}" for i in range(30)],
                                                       name="gene")), {})
    fp.make_subset("fetal100", 100)
    fp.make_subset("fetal200", 200)
    small = ad.read_h5ad(tmp_path / "fetal100.h5ad")
    big = ad.read_h5ad(tmp_path / "fetal200.h5ad")
    assert small.shape == (100, 30) and big.shape == (200, 30)
    assert set(small.obs_names) <= set(big.obs_names), "subsets must be nested"
    # every kept row is that cell's row of the parent, unchanged
    idx = [int(n[1:]) for n in small.obs_names]
    assert abs(small.X - full[idx]).max() == 0
    assert list(small.obs_names) == sorted(small.obs_names, key=lambda n: int(n[1:]))


# ------------------------------------------------------------ comparison


def test_published_markers_are_the_expressed_specific_ones() -> None:
    if not fb.DE.exists():
        pytest.skip("GSE156793_S8_DE_gene_cells.csv.gz not downloaded")
    m = fb.published_markers()
    assert len(m) == 77
    assert {"ALB", "AFP"} <= set(m["Hepatoblasts"])
    assert any(g.startswith("HB") for g in m["Erythroblasts"])
    assert all(len(v) == fb.N_PUBLISHED_MARKERS for v in m.values())


def _fake_run(dirpath: Path, labels, cells, markers) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / "labels.npy", np.asarray(labels, dtype=np.uint32))
    np.savetxt(dirpath / "obs_names.txt", np.asarray(cells, dtype=object), fmt="%s")
    markers.to_csv(dirpath / "markers.csv", index=False)


def test_compare_scores_a_perfect_clustering(tmp_path: Path, monkeypatch, capsys) -> None:
    """Clusters that are exactly the published types: agreement 1, every type
    recovered, and each cluster's markers are its own type's."""
    types = ["Hepatoblasts", "Erythroblasts", "Microglia"]
    cells = [f"cell{i}" for i in range(300)]
    labels = np.repeat([0, 1, 2], 100)
    pub = np.repeat(types, 100)
    pd.DataFrame({"cell": cells, "Main_cluster_name": pub,
                  "Organ": np.repeat(["Liver", "Liver", "Cerebrum"], 100)}
                 ).to_csv(tmp_path / "cells.csv.gz", index=False)
    marker_table = fb.published_markers()
    rows = [{"cluster": c, "gene": g, "scores": 10.0 - i}
            for c, t in enumerate(types)
            for i, g in enumerate(marker_table[t][:fb.N_OUR_MARKERS])]
    monkeypatch.setattr(fb, "OUT", tmp_path)
    monkeypatch.setattr(fb, "CELLS", tmp_path / "cells.csv.gz")
    _fake_run(tmp_path / "test", labels, cells, pd.DataFrame(rows))

    fb.run_compare("test")
    res = json.loads((tmp_path / "test" / "comparison.json").read_text())
    assert res["ari"] == 1.0 and res["nmi"] == 1.0
    assert res["types_recovered"] == 3
    assert res["cells_in_cluster_of_own_type"] == 1.0
    assert res["marker_overlap_mean"] == fb.N_OUR_MARKERS
    assert res["marker_overlap_other_types_mean"] < 1.0
    per_type = pd.read_csv(tmp_path / "test" / "per_type.csv", index_col=0)
    assert set(per_type.index) == set(types)
    assert (per_type["recall"] == 1.0).all()


def test_compare_reports_a_merged_type(tmp_path: Path, monkeypatch) -> None:
    """Two published types in one cluster: the smaller is not recovered, and
    the cluster is scored against the type that dominates it."""
    cells = [f"cell{i}" for i in range(300)]
    pub = np.array(["Hepatoblasts"] * 200 + ["Microglia"] * 100)
    labels = np.zeros(300, dtype=int)          # everything in one cluster
    pd.DataFrame({"cell": cells, "Main_cluster_name": pub,
                  "Organ": ["Liver"] * 300}).to_csv(tmp_path / "cells.csv.gz", index=False)
    rows = [{"cluster": 0, "gene": g, "scores": 1.0}
            for g in fb.published_markers()["Hepatoblasts"][:fb.N_OUR_MARKERS]]
    monkeypatch.setattr(fb, "OUT", tmp_path)
    monkeypatch.setattr(fb, "CELLS", tmp_path / "cells.csv.gz")
    _fake_run(tmp_path / "test", labels, cells, pd.DataFrame(rows))

    fb.run_compare("test")
    res = json.loads((tmp_path / "test" / "comparison.json").read_text())
    per_type = pd.read_csv(tmp_path / "test" / "per_type.csv", index_col=0)
    assert bool(per_type.loc["Hepatoblasts", "recovered"]) is True
    assert bool(per_type.loc["Microglia", "recovered"]) is False   # purity 2/3
    assert per_type.loc["Microglia", "recall"] == 1.0
    assert res["types_recovered"] == 1
    assert res["cells_in_cluster_of_own_type"] == pytest.approx(200 / 300, abs=1e-3)


def test_compare_counts_a_split_type_as_recovered(tmp_path: Path, monkeypatch) -> None:
    """A type spread over two clusters that are each mostly it is recovered
    biology: the strict rule misses it, the split-tolerant one does not."""
    cells = [f"cell{i}" for i in range(300)]
    pub = np.array(["Hepatoblasts"] * 200 + ["Microglia"] * 100)
    labels = np.array([0] * 70 + [1] * 70 + [2] * 60 + [3] * 100)   # the type in three
    pd.DataFrame({"cell": cells, "Main_cluster_name": pub,
                  "Organ": ["Liver"] * 300}).to_csv(tmp_path / "cells.csv.gz", index=False)
    rows = [{"cluster": c, "gene": g, "scores": 1.0} for c in (0, 1, 2)
            for g in fb.published_markers()["Hepatoblasts"][:fb.N_OUR_MARKERS]]
    monkeypatch.setattr(fb, "OUT", tmp_path)
    monkeypatch.setattr(fb, "CELLS", tmp_path / "cells.csv.gz")
    _fake_run(tmp_path / "test", labels, cells, pd.DataFrame(rows))

    fb.run_compare("test")
    per_type = pd.read_csv(tmp_path / "test" / "per_type.csv", index_col=0)
    h = per_type.loc["Hepatoblasts"]
    assert h["recall"] == 0.35 and not h["recovered"]    # no one cluster holds half
    assert h["clusters_of_this_type"] == 3
    assert h["recall_split"] == 1.0 and h["purity_split"] == 1.0 and h["recovered_split"]
    res = json.loads((tmp_path / "test" / "comparison.json").read_text())
    assert res["types_recovered"] == 1 and res["types_recovered_allowing_splits"] == 2
