"""Paris against its references: scikit-network's dendrogram, bit for bit;
its straight cut, as a partition; Scarf's balanced cut, label for label.

The references (tests/data/paris_reference.npz) were computed with
scikit-network and Scarf in the Scarf environment (NumPy 1.26), so the port is run with NumPy 1.x
summation (``numpy_sum_block=8192``); that only moves the last bits of
two normalising totals, which is exactly what a bit-for-bit test must pin.
"""
from pathlib import Path

import numpy as np
import pytest
from polariseq._polariseq import (
    dendrogram_cut_balanced,
    dendrogram_cut_straight,
    paris_dendrogram,
)

REF_PATH = Path(__file__).resolve().parents[3] / "tests/data/paris_reference.npz"
REF = np.load(REF_PATH) if REF_PATH.exists() else None
GRAPHS = [str(g) for g in REF["graphs"]] if REF is not None else []
needs_ref = pytest.mark.skipif(REF is None, reason="tests/data/paris_reference.npz not found")


def _paris(g: str, reorder: bool = False) -> np.ndarray:
    return paris_dendrogram(
        REF[f"{g}_indptr"].astype(np.uint32),
        REF[f"{g}_indices"].astype(np.uint32),
        REF[f"{g}_data"],
        int(REF[f"{g}_n"]),
        reorder=reorder,
        numpy_sum_block=8192,
    )


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@needs_ref
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("g", GRAPHS)
def test_dendrogram_is_bit_identical_to_scikit_network(g, reorder):
    got = _paris(g, reorder)
    want = REF[f"{g}_paris_reordered" if reorder else f"{g}_paris"]
    assert got.shape == want.shape
    # Exact: merge order, children, heights (to the last bit) and sizes.
    np.testing.assert_array_equal(got, want)


@needs_ref
@pytest.mark.parametrize("g", GRAPHS)
def test_straight_cut_gives_the_reference_partition(g):
    d = REF[f"{g}_paris"]
    for k in REF[f"{g}_ks"]:
        got = dendrogram_cut_straight(d, int(k))
        want = REF[f"{g}_straight_{int(k)}"]
        assert _same_partition(got, want), f"{g}, k={k}"


@needs_ref
@pytest.mark.parametrize("g", GRAPHS)
def test_balanced_cut_gives_scarfs_labels(g):
    d = REF[f"{g}_paris"].copy()
    d[d == np.inf] = 0  # as Scarf passes it
    for i in REF[f"{g}_balanced"]:
        mx, mn, fc = REF["balanced_params"][int(i)]
        got = dendrogram_cut_balanced(d, int(mx), int(mn), float(fc), numpy_sum_block=8192)
        np.testing.assert_array_equal(got, REF[f"{g}_balanced_{int(i)}"], err_msg=f"{g}, params {i}")


def test_tl_paris_stores_reuses_and_cuts():
    import polariseq as ps

    rng = np.random.default_rng(0)
    centres = rng.normal(0, 4, (6, 20))
    x = (centres[rng.integers(0, 6, 1500)] + rng.normal(0, 1, (1500, 20))).astype(np.float32)
    a = ps.AnnData(X=x)
    a.obsm["X_pca"] = x
    ps.pp.neighbors(a, n_neighbors=15, approximate=False)
    ps.tl.paris(a, n_clusters=6)
    d = a.uns["paris"]["dendrogram"]
    assert d.shape == (a.n_obs - 1, 4)
    assert d[-1, 3] == a.n_obs
    labels = np.asarray(a.obs["paris"])
    assert labels.min() == 0 and len(np.unique(labels)) >= 6
    # Well-separated groups: the 6-cluster cut recovers them.
    from sklearn.metrics import adjusted_rand_score

    truth = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    assert adjusted_rand_score(truth, labels) > 0.95
    # A second cut reuses the dendrogram (same object).
    ps.tl.paris(a, n_clusters=3)
    assert a.uns["paris"]["dendrogram"] is d
    assert len(np.unique(a.obs["paris"])) >= 3
    ps.tl.paris(a, balanced=True, min_size=20, max_size=400)
    sizes = np.bincount(np.asarray(a.obs["paris"]))
    assert sizes.max() <= 400 and np.asarray(a.obs["paris"]).min() == 0
    with pytest.raises(ValueError):
        ps.tl.paris(a, balanced=True)
