"""CSC-encoded `.h5ad` row reads.

The lazy row reader (`read_h5ad_rows`, and with it `head()`, `sample()` and
the whole disk-backed path) used to slice `X/indptr` as row pointers, which
is only valid for CSR. On a csc-encoded file — pbmc3k as shipped by scanpy —
it returned garbage rows, which surfaced as the spill arm's QC keeping 256
of 2,700 cells (found by the telemetry harness, chapter 09). CSC row ranges
are not contiguous on disk, so the reader now does one full read plus a
counting-sort transpose and selects rows in memory — correct over fast.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest
import scipy.sparse as sp

scipy_h5py = pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def csc_file(tmp_path_factory):
    """A small csc-encoded h5ad with known content, written like AnnData does."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(40, 30)).astype(np.float32)
    counts[rng.random((40, 30)) < 0.6] = 0
    csc = sp.csc_matrix(counts)
    path = tmp_path_factory.mktemp("csc") / "csc_40x30.h5ad"
    with scipy_h5py.File(path, "w") as h:
        g = h.create_group("X")
        g.attrs["encoding-type"] = "csc_matrix"
        g.attrs["shape"] = csc.shape
        g.create_dataset("data", data=csc.data)
        g.create_dataset("indices", data=csc.indices.astype(np.uint32))
        g.create_dataset("indptr", data=csc.indptr.astype(np.uint32))
    return str(path), counts


class TestCscRowReads:
    def test_row_slice_matches_the_dense_matrix(self, csc_file) -> None:
        path, counts = csc_file
        r = ps._polariseq.read_h5ad_rows(path, 5, 12)
        got = np.zeros((7, counts.shape[1]), dtype=np.float32)
        rows, cols = np.nonzero(counts[5:12])
        for i, j in zip(rows, cols, strict=True):
            got[i, j] = counts[5 + i, j]
        assert r["n_cells"] == 7
        assert r["n_genes"] == counts.shape[1]
        np.testing.assert_array_equal(
            sp.csr_matrix(
                (r["data"], r["indices"], r["indptr"]), shape=(7, counts.shape[1])
            ).toarray(),
            got,
        )

    def test_disk_backed_path_filters_correctly_on_csc(self, csc_file) -> None:
        """The end-to-end consequence: QC on a csc file must see the real
        per-cell gene counts, so min_genes keeps the cells it should."""
        path = _csc_with_obs(csc_file)
        _, counts = csc_file
        a = ps.process_diskbacked(path, n_hvg=10, n_pcs=3, n_neighbors=5,
                                  seed=0, min_genes=8)
        expected = int((np.count_nonzero(counts, axis=1) >= 8).sum())
        assert expected > 0
        assert a.n_obs == expected, (
            "the disk path's QC misread the csc matrix (pre-fix it kept a "
            "near-arbitrary subset)"
        )


def _csc_with_obs(csc_file) -> str:
    """The disk pipeline wants obs/var names in the file; add minimal ones."""
    import shutil

    path, counts = csc_file
    out = path.replace(".h5ad", "_obs.h5ad")
    shutil.copyfile(path, out)
    with scipy_h5py.File(out, "a") as h:
        obs = h.create_group("obs")
        obs.create_dataset("_index", data=np.array([f"c{i}" for i in range(40)], dtype="S8"))
        var = h.create_group("var")
        var.create_dataset("_index", data=np.array([f"g{i}" for i in range(30)], dtype="S8"))
    return out
