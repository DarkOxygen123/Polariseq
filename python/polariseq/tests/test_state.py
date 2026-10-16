"""Saved analyses: ``adata.save(name)`` and ``ps.load(project, name)``.

A save points to the counts instead of storing them, links the results the
project already holds, and a load gives back the analysis as it was: the same
matrix to the last bit (in memory rebuilt from the files, out of core the
on-disk copy with its view), the same annotations and results, ready for the
next step.
"""

from __future__ import annotations

import gc
import json
import shutil
import warnings

import numpy as np
import polariseq as ps
import polariseq._project as P
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def analyze(a, *, renormalize: bool = False):
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=120)
    ps.pp.filter_genes(a, min_cells=300)
    ps.pp.normalize_total(a, target_sum=1e4)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=300, subset=True)
    if renormalize:
        ps.pp.renormalize(a)
    ps.pp.pca(a, 15, seed=0)
    ps.pp.neighbors(a, n_neighbors=10, seed=0)
    ps.tl.leiden(a, seed=0)
    return a


def matrix(x):
    """The values of a matrix, whatever holds it, as a SciPy CSR matrix."""
    if isinstance(x, ps.DiskMatrix):
        return sp.csr_matrix(x[:, list(range(x.shape[1]))])  # read by columns
    return x.to_scipy()


def same(a, b) -> None:
    """Two analyses agree in every part."""
    assert a.shape == b.shape
    assert list(a.obs_names) == list(b.obs_names) and list(a.var_names) == list(b.var_names)
    assert set(a.obs) == set(b.obs) and set(a.obsm) == set(b.obsm) and set(a.obsp) == set(b.obsp)
    for k in a.obs:
        np.testing.assert_array_equal(np.asarray(a.obs[k]), np.asarray(b.obs[k]), err_msg=k)
    for k in a.obsm:
        np.testing.assert_array_equal(np.asarray(a.obsm[k]), np.asarray(b.obsm[k]), err_msg=k)
    for k in a.obsp:
        assert (a.obsp[k] != b.obsp[k]).nnz == 0, k
    assert set(a.uns) == set(b.uns)
    if isinstance(a.X, ps.DiskMatrix):
        np.testing.assert_array_equal(a.X.row_mask(), b.X.row_mask())
        np.testing.assert_array_equal(a.X.col_mask(), b.X.col_mask())
        assert a.X.transforms() == b.X.transforms()
        if a.X.scale_factors() is not None:
            np.testing.assert_array_equal(a.X.scale_factors(), b.X.scale_factors())
    x, y = matrix(a.X), matrix(b.X)
    np.testing.assert_array_equal(x.indptr, y.indptr)
    np.testing.assert_array_equal(x.indices, y.indices)
    np.testing.assert_array_equal(x.data, y.data)  # bit for bit


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "data" / "cells.h5ad"
    p.parent.mkdir()
    shutil.copy(FIXTURE, p)
    return p


@pytest.mark.parametrize("renormalize", [False, True])
def test_in_memory_the_matrix_is_rebuilt_not_stored(proj, source, renormalize) -> None:
    a = analyze(ps.read_h5ad(str(source), mode="memory"), renormalize=renormalize)
    folder = a.save("clustered")
    m = json.loads((folder / "manifest.json").read_text())
    assert m["matrix"]["kind"] == "rebuild"
    stored = sum(f.stat().st_size for f in folder.glob("*.npy"))
    assert stored < source.stat().st_size / 2  # the counts are not in it
    b = ps.load(proj.path, "clustered")
    same(a, b)
    # The next step runs on the loaded analysis as on the one in hand.
    ps.tl.leiden(a, resolution=0.5, seed=1, key_added="fine")
    ps.tl.leiden(b, resolution=0.5, seed=1, key_added="fine")
    np.testing.assert_array_equal(np.asarray(a.obs["fine"]), np.asarray(b.obs["fine"]))


def test_out_of_core_the_copy_is_reopened_with_its_view(proj, source) -> None:
    a = analyze(ps.read_h5ad(str(source), mode="out-of-core"), renormalize=True)
    a.save("ooc")
    m = json.loads((proj.path / "states" / "ooc" / "manifest.json").read_text())
    assert m["matrix"]["kind"] == "store"
    b = ps.load(proj.path, "ooc")
    assert b.mode == "out-of-core" and isinstance(b.X, ps.DiskMatrix)
    same(a, b)
    ps.tl.rank_genes_groups(a, groupby="leiden")
    ps.tl.rank_genes_groups(b, groupby="leiden")
    assert ps.get.markers(a, n=3).equals(ps.get.markers(b, n=3))


def test_a_deleted_copy_is_made_again(proj, source) -> None:
    a = analyze(ps.read_h5ad(str(source), mode="out-of-core"))
    a.save("ooc")
    proj.clean("matrix")
    same(a, ps.load(proj.path, "ooc"))


def test_several_files_in_memory(proj, tmp_path) -> None:
    full = ad.read_h5ad(FIXTURE)
    full.X = sp.csr_matrix(full.X)
    paths = {}
    for k, part in {"d1": full[:6000], "d2": full[6000:, ::-1]}.items():
        paths[k] = str(tmp_path / f"{k}.h5ad")
        part.copy().write_h5ad(paths[k])
    a = analyze(ps.read_h5ad(paths, mode="memory"))
    a.save("both")
    same(a, ps.load(proj.path, "both"))


def test_a_moved_source_stops_the_load_and_relink_finds_it(proj, source, tmp_path) -> None:
    a = analyze(ps.read_h5ad(str(source), mode="memory"))
    a.save("s")
    moved = tmp_path / "elsewhere.h5ad"
    source.rename(moved)
    with pytest.raises(ps.SourceMovedError, match="no longer there") as e:
        ps.load(proj.path, "s")
    assert str(source) in str(e.value) and "relink" in str(e.value)
    b = ps.load(proj.path, "s", relink={str(source): str(moved)})
    same(a, b)
    assert b._source["paths"] == [str(moved)]


def test_a_project_moved_with_its_data_keeps_loading(tmp_path) -> None:
    before = ps.budget()
    try:
        root = tmp_path / "proj"
        ps.project(root)
        (root / "data").mkdir()
        shutil.copy(FIXTURE, root / "data" / "cells.h5ad")
        a = analyze(ps.read_h5ad(str(root / "data" / "cells.h5ad"), mode="memory"))
        a.save("s")
        # what to compare, in memory: Windows cannot move a folder whose files are still mapped
        want_x, want_pca = a.X.to_scipy(), np.array(a.obsm["X_pca"])
        want_leiden = np.array(a.obs["leiden"])
        del a
        gc.collect()
        P._CURRENT = None
        shutil.move(root, tmp_path / "moved")
        b = ps.load(tmp_path / "moved", "s")
        np.testing.assert_array_equal(b.X.to_scipy().data, want_x.data)
        np.testing.assert_array_equal(np.asarray(b.obsm["X_pca"]), want_pca)
        np.testing.assert_array_equal(np.asarray(b.obs["leiden"]), want_leiden)
    finally:
        P._CURRENT = None
        ps.set_budget(ram_gb=before.ram_gb, spill_dir=before.spill_dir)


def test_results_are_linked_and_outlive_a_rerun(proj, source) -> None:
    a = ps.read_h5ad(str(source), mode="memory")
    ps.pp.scrublet(a)
    score = a.obs["doublet_score"]
    a.save("s")
    linked = [f for f in (proj.path / "states" / "s").glob("*.npy") if f.stat().st_nlink > 1]
    assert linked  # the scores' file, not a copy
    ps.pp.scrublet(a, random_state=1)  # replaces the step's own file
    b = ps.load(proj.path, "s")
    np.testing.assert_array_equal(np.asarray(b.obs["doublet_score"]), np.asarray(score))


def test_a_result_changed_in_memory_is_written(proj, source) -> None:
    a = ps.read_h5ad(str(source), mode="memory")
    ps.pp.scrublet(a)
    a.obs["doublet_score"][0] = 99.0  # a copy-on-write view: the file keeps its value
    a.save("s")
    assert float(ps.load(proj.path, "s").obs["doublet_score"][0]) == 99.0


def test_a_matrix_set_by_hand_is_stored(proj, source) -> None:
    a = ps.read_h5ad(str(source), mode="memory")
    ps.pp.normalize_total(a)
    a.X = sp.csr_matrix(a.X.to_scipy() * 2)  # not a recorded step
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folder = a.save("s")  # nothing to warn about: no record, so stored
    assert json.loads((folder / "manifest.json").read_text())["matrix"]["kind"] == "stored-csr"
    same(a, ps.load(proj.path, "s"))


def test_a_source_changed_since_reading_is_noticed_at_save(proj, source) -> None:
    a = ps.read_h5ad(str(source), mode="memory")
    ps.pp.normalize_total(a)
    other = ad.read_h5ad(FIXTURE)
    other.X = sp.csr_matrix(other.X) * 3
    other.write_h5ad(source)
    with pytest.warns(UserWarning, match="does not follow from its source files"):
        folder = a.save("s")
    assert json.loads((folder / "manifest.json").read_text())["matrix"]["kind"] == "stored-csr"


def test_tables_and_annotations_round_trip(proj, source) -> None:
    a = analyze(ps.read_h5ad(str(source), mode="memory"))
    a.obs["group"] = np.where(np.arange(a.n_obs) % 2 == 0, "ctrl", "stim").astype(object)
    a.obs["donor"] = pd.Categorical(np.arange(a.n_obs) % 3)
    a.uns["note"] = {"by": "me", 3: [1.5, None, np.int32(4)], "t": (1, "x")}
    a.uns["table"] = pd.DataFrame({"gene": ["a", "b"], "lfc": [1.0, np.nan],
                                   "kind": pd.Categorical(["x", "y"])})
    a.save("s")
    b = ps.load(proj.path, "s")
    assert b.uns["note"] == {"by": "me", 3: [1.5, None, np.int32(4)], "t": (1, "x")}
    pd.testing.assert_frame_equal(b.uns["table"], a.uns["table"])
    assert list(b.obs["group"][:2]) == ["ctrl", "stim"]
    assert b.obs["donor"].categories.tolist() == [0, 1, 2]


def test_names_and_overwrite(proj, source) -> None:
    a = ps.read_h5ad(str(source), mode="memory")
    with pytest.raises(ValueError, match="letters"):
        a.save("../x")
    a.save("s")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        a.save("s")
    a.save("s", overwrite=True)
    with pytest.raises(FileNotFoundError, match="saved: s"):
        ps.load(proj.path, "t")


def test_gene_subset_keeps_the_source_genes(proj, source) -> None:
    """``highly_variable_genes(subset=True)`` records which source genes remain,
    as the gene filter does, so raw counts are read back for the right genes."""
    a = ps.read_h5ad(str(source), mode="memory")
    ps.pp.filter_genes(a, min_cells=300)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    ps.pp.highly_variable_genes(a, n_top_genes=100, subset=True)
    assert len(a._cols) == a.n_vars == 100
    a.obs["g"] = pd.Categorical(np.arange(a.n_obs) % 2)
    pb = ps.get.pseudobulk(a, "g")
    raw = ad.read_h5ad(FIXTURE)
    want = np.asarray(raw[:, list(a.var_names)].X.sum(axis=0)).ravel()
    np.testing.assert_allclose(np.asarray(pb.X).sum(axis=0), want, rtol=1e-5)
