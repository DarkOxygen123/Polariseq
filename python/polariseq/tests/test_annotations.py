"""Cell annotations travel with the cells.

Every column encoding anndata writes is read into ``obs``, in memory and out
of core; labels stay categorical (one small integer per cell) through reading,
filtering and export; and several files read as one have their columns joined
the way ``anndata.concat`` joins them.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")


def _file(path, n: int, seed: int, *, donors: list[str], extra: bool) -> str:
    rng = np.random.default_rng(seed)
    x = sp.random(n, 30, density=0.3, format="csr", dtype=np.float32, random_state=seed)
    x.data = np.ceil(x.data * 9)
    obs = pd.DataFrame({
        "donor": pd.Categorical(rng.choice(donors, n), categories=donors),
        "sample_id": [f"s{seed}_{i % 4}" for i in range(n)],
        "score": rng.normal(size=n).astype(np.float32),
        "flag": rng.random(n) > 0.5,
        "count": pd.arrays.IntegerArray(rng.integers(0, 9, n), rng.random(n) < 0.2),
    }, index=[f"c{seed}_{i}" for i in range(n)])
    if extra:
        obs["stage"] = pd.Categorical(rng.choice(["early", "late"], n),
                                      categories=["early", "late"], ordered=True)
    a = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(
        {"gene_ids": [f"ENSG{j:05d}" for j in range(30)]}, index=[f"g{j}" for j in range(30)]))
    a.write_h5ad(path)
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> list[str]:
    d = tmp_path_factory.mktemp("annotations")
    return [_file(d / "ctrl.h5ad", 200, 1, donors=["d1", "d2", "d3"], extra=True),
            _file(d / "stim.h5ad", 150, 2, donors=["d3", "d1", "d4"], extra=False)]


def _same(got, want) -> None:
    """Equal values, missing where missing."""
    g, w = pd.Series(np.asarray(got, dtype=object)), pd.Series(np.asarray(want, dtype=object))
    assert (g.isna() == w.isna()).all()
    assert (g[~g.isna()].astype(str) == w[~w.isna()].astype(str)).all()


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_one_file_reads_every_column(files, mode, proj):
    a = ps.read_h5ad(files[0], mode=mode)
    want = ad.read_h5ad(files[0]).obs
    assert list(a.obs) == list(want.columns)
    assert isinstance(a.obs["donor"], pd.Categorical)
    assert a.obs["donor"].codes.dtype == np.int8
    # Text arrives as codes into its distinct values, not a string per cell.
    assert isinstance(a.obs["sample_id"], pd.Categorical)
    assert a.obs["stage"].ordered
    assert a.obs["score"].dtype == np.float32
    assert a.obs["flag"].dtype == np.bool_
    for c in want.columns:
        _same(a.obs[c], want[c])
    assert list(a.var) == ["gene_ids"]


def test_several_files_join_their_columns_as_anndata_concat(files):
    a = ps.read_h5ad({"ctrl": files[0], "stim": files[1]}, batch_key="condition")
    want = ad.concat([ad.read_h5ad(p) for p in files], join="outer", label="condition",
                     keys=["ctrl", "stim"])
    assert set(a.obs) == set(want.obs.columns)
    for c in want.obs.columns:
        _same(a.obs[c], want.obs[c])
    # Categories are the union, in order of first appearance, codes compact.
    assert list(a.obs["donor"].categories) == ["d1", "d2", "d3", "d4"]
    assert a.obs["condition"].codes.dtype == np.int8
    # A column only the first file has is missing for the second file's cells.
    assert a.obs["stage"].isna().sum() == 150


def test_obs_columns_limits_the_read(files):
    a = ps.read_h5ad(files[0], obs_columns=["donor"])
    assert list(a.obs) == ["donor"]
    b = ps.read_h5ad(files, obs_columns=["stage"])
    assert list(b.obs) == ["stage", "dataset"]
    with pytest.raises(KeyError, match="nope"):
        ps.read_h5ad(files, obs_columns=["nope"])
    with pytest.raises(RuntimeError, match="nope"):
        ps.read_h5ad(files[0], obs_columns=["nope"])


def test_labels_stay_categorical_through_filtering_and_export(files):
    a = ps.read_h5ad({"ctrl": files[0], "stim": files[1]}, batch_key="condition")
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_cells(a, min_genes=9)
    kept = a.n_obs
    assert kept < 350
    assert isinstance(a.obs["donor"], pd.Categorical)
    assert len(a.obs["donor"]) == kept
    out = a.to_anndata()
    assert isinstance(out.obs["donor"].dtype, pd.CategoricalDtype)
    assert list(out.obs["condition"].cat.categories) == ["ctrl", "stim"]
