"""Genes kept whatever the ranking chooses: ``include=`` in gene selection.

The genes asked for are added to the top ``n_top_genes`` (only those not
already among them), in memory, out of core and in the one-call pipeline; the
steps after selection run on them; a gene that is not there is an error that
says why.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

FIXTURE = "tests/data/medium_10kx2k.h5ad"


def prepared(mode: str):
    a = ps.read_h5ad(FIXTURE, mode=mode)
    ps.pp.calculate_qc_metrics(a)
    ps.pp.filter_genes(a, min_cells=50)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    return a


def ranked_out(a, n: int, k: int) -> list[str]:
    """``k`` genes the ranking leaves out, detected in enough cells."""
    b = prepared("memory")
    ps.pp.highly_variable_genes(b, n_top_genes=n)
    hv = np.asarray(b.var["highly_variable"])
    return [str(g) for g, h in zip(b.var_names, hv, strict=True) if not h][:k]


@pytest.mark.parametrize("mode", ["memory", "out-of-core"])
def test_genes_are_added_to_the_top_genes(proj, mode) -> None:
    plain = prepared(mode)
    ps.pp.highly_variable_genes(plain, n_top_genes=200)
    top = [str(g) for g, h in zip(plain.var_names, plain.var["highly_variable"], strict=True) if h]
    extra = ranked_out(plain, 200, 3)
    a = prepared(mode)
    ps.pp.highly_variable_genes(a, n_top_genes=200, include=[top[0], *extra], subset=True)
    assert a.n_vars == 203  # the gene already chosen is not counted twice
    assert set(top) | set(extra) == {str(g) for g in a.var_names}
    assert int(np.sum(a.var["included"])) == 4
    assert a.uns["hvg"]["include_added"] == extra
    # The steps after selection run on them.
    ps.pp.renormalize(a)
    ps.pp.pca(a, 10, seed=0)
    assert a.obsm["X_pca"].shape == (a.n_obs, 10)


def test_without_subset_the_mask_holds_them(proj) -> None:
    a = prepared("memory")
    extra = ranked_out(a, 100, 2)
    ps.pp.highly_variable_genes(a, n_top_genes=100, include=extra)
    hv = dict(zip(map(str, a.var_names), np.asarray(a.var["highly_variable"]), strict=True))
    assert all(hv[g] for g in extra) and sum(hv.values()) == 102


def test_a_gene_not_in_the_data_is_a_clear_error(proj) -> None:
    a = prepared("memory")
    with pytest.raises(KeyError) as e:
        ps.pp.highly_variable_genes(a, n_top_genes=100, include=["gene_12", "gene_x12", "GENE_13"])
    msg = str(e.value)
    assert "2 of the 3 genes" in msg and "gene_x12: not in the data" in msg
    assert "Did you mean 'gene_13'?" in msg  # a different case found


def test_a_gene_removed_by_the_filter_says_so(proj) -> None:
    a = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.calculate_qc_metrics(a)
    n_cells = np.asarray(a.var["n_cells"])
    rare = str(a.var_names[int(np.argmin(n_cells))])
    ps.pp.filter_genes(a, min_cells=int(n_cells.min()) + 1)
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    with pytest.raises(KeyError, match=f"{rare}: in the data, but removed by an earlier step"):
        ps.pp.highly_variable_genes(a, n_top_genes=100, include=[rare])


def test_a_rare_gene_is_kept_with_a_warning(proj) -> None:
    a = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.calculate_qc_metrics(a)
    rare = str(a.var_names[7])
    a.var["n_cells"][7] = 3  # as the gene filter would count a gene seen in three cells
    ps.pp.normalize_total(a)
    ps.pp.log1p(a)
    with pytest.warns(UserWarning, match=rf"include: detected in .* {rare} \(3 cells, 0\.03%\)"):
        ps.pp.highly_variable_genes(a, n_top_genes=100, include=[rare], subset=True)
    assert rare in list(a.var_names)


def test_the_one_call_pipeline(proj) -> None:
    ps.set_budget(ram_gb=2)
    extra = ranked_out(None, 300, 2)
    a = ps.process_diskbacked(FIXTURE, n_hvg=300, n_pcs=10, n_neighbors=10, min_cells=50,
                              include=extra)
    names = [str(g) for g in a.var["name"]]
    hv = dict(zip(names, np.asarray(a.var["highly_variable"]), strict=True))
    assert all(hv[g] for g in extra)
    assert a.uns["hvg"]["include"] == extra
    with pytest.raises(KeyError, match="gene_x1: not in the data"):
        ps.process_diskbacked(FIXTURE, n_hvg=300, include=["gene_x1"])


def test_the_one_call_pipeline_names_a_gene_its_filter_removes(proj) -> None:
    ps.set_budget(ram_gb=2)
    a = ps.read_h5ad(FIXTURE, mode="memory")
    ps.pp.calculate_qc_metrics(a)
    n_cells = np.asarray(a.var["n_cells"])
    rare = str(a.var_names[int(np.argmin(n_cells))])
    with pytest.raises(ValueError, match=rf"min_cells=\d+, so the gene filter removes them: {rare}"):
        ps.process_diskbacked(FIXTURE, n_hvg=300, min_cells=int(n_cells.min()) + 1, include=[rare])


def test_a_saved_analysis_keeps_them(proj) -> None:
    a = prepared("memory")
    extra = ranked_out(a, 150, 2)
    ps.pp.highly_variable_genes(a, n_top_genes=150, include=extra, subset=True)
    a.save("with_genes")
    b = ps.load(proj.path, "with_genes")
    assert list(b.var_names) == list(a.var_names)
    np.testing.assert_array_equal(b.X.to_scipy().data, a.X.to_scipy().data)
