"""Files past 2^32 stored values are read where their rows lie.

A matrix with more than 4,294,967,295 stored values (an atlas of tens of
millions of cells) needs 64-bit row pointers. The out-of-core reader takes
them at full width and hands each block 32-bit offsets of its own. The file
here is small on disk: its datasets are chunked and only the chunk holding
the rows read is written; the rest is an unwritten hole of 2^32 + 3 values,
all in row 0, which is never read.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from polariseq._polariseq import read_h5ad_rows  # noqa: E402

HOLE = (1 << 32) + 3


def test_rows_past_two_to_the_32_values(tmp_path) -> None:
    path = tmp_path / "wide.h5ad"
    vals = np.array([1.5, 2.0, 3.25, 4.0, 5.5], dtype=np.float32)
    cols = np.array([0, 3, 2, 1, 4], dtype=np.int32)
    with h5py.File(path, "w") as f:
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.array([4, 5], dtype=np.int64)
        n = HOLE + len(vals)
        d = x.create_dataset("data", shape=(n,), dtype="f4", chunks=(1 << 20,))
        i = x.create_dataset("indices", shape=(n,), dtype="i4", chunks=(1 << 20,))
        d[HOLE:] = vals
        i[HOLE:] = cols
        x.create_dataset("indptr", data=np.array([0, HOLE, HOLE + 2, HOLE + 3, HOLE + 5],
                                                 dtype=np.int64))
    assert path.stat().st_size < 50_000_000  # the hole is not on disk
    rows = read_h5ad_rows(str(path), 1, 4)
    np.testing.assert_array_equal(rows["data"], vals)
    np.testing.assert_array_equal(rows["indices"], cols.astype(np.uint32))
    np.testing.assert_array_equal(rows["indptr"], [0, 2, 3, 5])
    one = read_h5ad_rows(str(path), 2, 3)
    np.testing.assert_array_equal(one["data"], vals[2:3])
