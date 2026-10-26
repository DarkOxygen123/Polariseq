"""Errors say what is missing or wrong, why, and what to do.

Each case is a typical mistake (a misspelled column, gene or file, a step run
before the one it needs, an impossible value), and the message must name the
thing, the closest correct name where there is one, and the remedy.
"""

from __future__ import annotations

import numpy as np
import polariseq as ps
import pytest

pd = pytest.importorskip("pandas")

FIXTURE = "tests/data/medium_10kx2k.h5ad"


@pytest.fixture(scope="module")
def stages(tmp_path_factory):
    """The fixture at each stage of the analysis, in memory."""
    def run(*steps):
        a = ps.read_h5ad(FIXTURE, mode="memory")
        ps.pp.calculate_qc_metrics(a)
        if "norm" in steps:
            ps.pp.normalize_total(a)
            ps.pp.log1p(a)
            ps.pp.highly_variable_genes(a, n_top_genes=200, subset=True)
        if "pca" in steps:
            ps.pp.pca(a, 10, seed=0)
        if "nn" in steps:
            ps.pp.neighbors(a, n_neighbors=10, seed=0)
        if "leiden" in steps:
            ps.tl.leiden(a, seed=0)
        a.obs["cond"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "ctrl", "stim"))
        a.obs["donor"] = pd.Categorical(np.arange(a.n_obs) % 4)
        return a

    return {"raw": run(), "norm": run("norm"), "pca": run("norm", "pca"),
            "full": run("norm", "pca", "nn", "leiden")}


CASES = [
    # (stage, call, exception, fragments of the message)
    (None, lambda a: ps.read_h5ad("no/such.h5ad"), FileNotFoundError, ["read_h5ad: no file no/such.h5ad"]),
    (None, lambda a: ps.read_h5ad("tests"), ValueError, ["is a folder"]),
    (None, lambda a: ps.read_h5ad("README.md"), ValueError, ["not an HDF5 file", "ps.read_10x_mtx"]),
    (None, lambda a: ps.plan("tests/data/medium_10kx2k.h5"), FileNotFoundError,
     ["no file", "Did you mean 'medium_10kx2k.h5ad'?"]),
    (None, lambda a: ps.set_budget(threads=0), ValueError, ["threads must be a positive whole number"]),
    ("norm", lambda a: ps.pp.neighbors(a), KeyError, ["adata.obsm['X_pca']", "ps.pp.pca makes"]),
    ("pca", lambda a: ps.pp.neighbors(a, use_rep="X_pca_harmony"), KeyError,
     ["ps.pp.harmony_integrate makes", "Present in adata.obsm: 'X_pca'"]),
    ("pca", lambda a: ps.pp.harmony_integrate(a, "donr"), KeyError, ["Did you mean 'donor'?"]),
    ("pca", lambda a: ps.tl.leiden(a), KeyError, ["neighbour graph", "ps.pp.neighbors"]),
    ("pca", lambda a: ps.tl.umap(a), KeyError, ["neighbour graph"]),
    ("norm", lambda a: ps.pp.pca(a, 5000), ValueError, ["at most 200 components"]),
    ("full", lambda a: ps.tl.rank_genes_groups(a, groupby="leidn"), KeyError, ["Did you mean 'leiden'?"]),
    ("full", lambda a: ps.tl.rank_genes_groups(a, groups=["999"]), KeyError, ["'999' not found in obs['leiden']"]),
    ("pca", lambda a: ps.tl.paga(a), KeyError, ["ps.pp.neighbors"]),
    ("full", lambda a: ps.tl.dpt(a, root=-5), ValueError, ["root=-5 is not a cell"]),
    ("full", lambda a: ps.tl.compare_conditions(a, "condd", "ctrl", sample_key="donor"), KeyError,
     ["condition_key='condd'", "Did you mean 'cond'?"]),
    ("full", lambda a: ps.tl.compare_conditions(a, "cond", "control", sample_key="donor"), KeyError,
     ["'control' not found in obs['cond']", "Did you mean 'ctrl'?"]),
    ("full", lambda a: ps.get.pseudobulk(a, "donr"), KeyError, ["sample_key='donr'"]),
    ("full", lambda a: ps.tl.score_genes(a, ["NOPE1"]), KeyError, ["NOPE1: not in the data"]),
    ("full", lambda a: ps.tl.annotate(a, {"999": "T"}), KeyError, ["'999' not found in obs['leiden']"]),
    ("full", lambda a: ps.get.markers(a), KeyError, ["run ps.tl.rank_genes_groups first"]),
    ("full", lambda a: a[:, ["gene_x"]], KeyError, ["no genes named 'gene_x'"]),
]


@pytest.mark.parametrize(("stage", "call", "exc", "fragments"), CASES,
                         ids=[f"case{i}" for i in range(len(CASES))])
def test_the_message_names_the_problem_and_the_remedy(stages, proj, stage, call, exc, fragments):
    with pytest.raises(exc) as e:
        call(stages[stage] if stage else None)
    msg = str(e.value)
    for f in fragments:
        assert f in msg, msg
    assert not msg.startswith(("'", '"')), "a KeyError should print as a sentence, not a quoted repr"


def test_a_partial_gene_list_warns_and_names_the_missing(stages) -> None:
    a = stages["full"]
    gene = str(a.var_names[0])
    with pytest.warns(UserWarning, match="1 of the 2 genes are not in the data .* 'NOPE'"):
        ps.tl.score_genes(a, [gene, "NOPE"], score_name="partial")
