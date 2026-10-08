"""The large-run example and its time-and-memory tracker keep working.

`examples/06_large_run.py` is what a user runs on a multi-million-cell dataset,
so it is checked here on the bundled fixture, where it takes seconds: the log,
the figures and the exported cells are written, and a budget the data cannot
meet is declined before anything runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def example():
    sys.path.insert(0, str(EXAMPLES))   # the example imports its tracker as `_resources`
    return load("large_run_example", "06_large_run.py")


def test_a_run_writes_its_log_figures_and_cells(proj, tmp_path) -> None:
    out = tmp_path / "out"
    code = example().main(["--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--umap"])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    steps = [s["step"] for s in report["steps"]]
    assert steps[0] == "open" and {"pca", "neighbors", "leiden", "umap"} <= set(steps)
    assert report["dataset"]["cells_in_file"] == 10_000
    assert report["peak_memory_gb"]["held_os"] > 0 and report["seconds_total"] > 0

    cells = pd.read_csv(out / "cells.tsv.gz", sep="\t")
    assert len(cells) == report["dataset"]["cells_after_qc"]
    assert {"cell", "n_genes", "n_counts", "leiden", "umap1", "umap2"} <= set(cells)
    for name in ("qc.png", "pca.png", "clusters.png", "umap.png", "resources.png",
                 "resources.csv"):
        assert (out / name).stat().st_size > 0, name
    assert report["machine"]["cpu"] is not None or "cores_logical" in report["machine"]
    summary = (out / "run_summary.md").read_text()
    assert "## Machine" in summary and "## Time and memory" in summary


def test_a_budget_the_data_cannot_meet_is_declined_before_anything_runs(proj, tmp_path) -> None:
    out = tmp_path / "out"
    code = example().main(["--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "0.001", "--mode", "out-of-core"])
    assert code == 2
    assert not (out / "report.json").exists()


def test_the_tracker_times_steps_and_records_what_they_held(tmp_path) -> None:
    sys.path.insert(0, str(EXAMPLES))
    tracker = load("resources_example", "_resources.py")
    with tracker.ResourceLog(interval=0.05, watch=tmp_path) as log:
        with log.step("work"):
            block = bytearray(50_000_000)   # a step that holds something
            block[0] = 1
        with log.step("idle"):
            pass
    summary = log.summary()
    assert [s["step"] for s in summary["steps"]] == ["work", "idle"]
    assert summary["steps"][0]["seconds"] > 0 and summary["peak_rss_gb_os"] > 0
    files = log.save(tmp_path / "log")
    assert files["csv"].stat().st_size > 0 and files["png"].stat().st_size > 0


def test_a_folder_of_shards_reads_as_one_dataset(proj, tmp_path) -> None:
    import anndata as ad

    whole = ad.read_h5ad(EXAMPLES.parent / "tests/data/medium_10kx2k.h5ad")
    shards = tmp_path / "shards"
    shards.mkdir()
    for k, rows in enumerate((slice(0, 5000), slice(5000, 10000))):
        part = whole[rows].copy()
        part.obs = part.obs[[]]                       # a lean shard: counts and the cell index
        part.write_h5ad(shards / f"{k}.h5ad", compression="gzip")
    out = tmp_path / "out"
    code = example().main([str(shards), "--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--min-genes", "100"])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert len(report["dataset"]["files"]) == 2
    assert report["dataset"]["cells_in_file"] == 10_000
    assert len(pd.read_csv(out / "cells.tsv.gz", sep="\t")) == report["dataset"]["cells_after_qc"]


def test_input_files_are_taken_in_natural_order_and_errors_are_clear(tmp_path) -> None:
    for n in (10, 2, 1):
        (tmp_path / f"{n}.h5ad").touch()
    files = example().resolve_inputs([str(tmp_path)])
    assert [Path(f).name for f in files] == ["1.h5ad", "2.h5ad", "10.h5ad"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no .h5ad files"):
        example().resolve_inputs([str(tmp_path / "empty")])
    with pytest.raises(SystemExit, match="no such .h5ad file"):
        example().resolve_inputs([str(tmp_path / "missing.h5ad")])


def test_exported_cells_name_their_file_and_carry_the_columns_asked_for(proj, tmp_path) -> None:
    """Cell names can repeat across files (a collection may number its cells
    from 0 in each), so the export names each cell's file and keeps columns of
    the input on request, still lined up with the cells that passed quality
    control."""
    import anndata as ad
    import numpy as np

    whole = ad.read_h5ad(EXAMPLES.parent / "tests/data/medium_10kx2k.h5ad")
    shards = tmp_path / "shards"
    shards.mkdir()
    for k, rows in enumerate((slice(0, 5000), slice(5000, 10000))):
        part = whole[rows].copy()
        ids = np.arange(rows.start, rows.stop)
        part.obs = pd.DataFrame({"row_id": ids, "cell_type": np.where(ids % 3 == 0, "B", "T")},
                                index=[str(i) for i in range(5000)])   # names repeat per file
        part.write_h5ad(shards / f"{k}.h5ad", compression="gzip")
    out = tmp_path / "out"
    code = example().main([str(shards), "--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--min-genes", "170",
                           "--keep-obs", "row_id, cell_type"])
    assert code == 0
    cells = pd.read_csv(out / "cells.tsv.gz", sep="\t")
    assert 0 < len(cells) < 10_000                       # quality control removed some cells
    assert cells["cell"].duplicated().any()              # the name alone does not identify a cell
    assert not cells.duplicated(["dataset", "cell"]).any()
    assert set(cells["dataset"].astype(str)) == {"0", "1"}
    counts = np.asarray(whole.X.sum(axis=1)).ravel()     # the kept cells are the ones named
    np.testing.assert_allclose(cells["n_counts"], counts[cells["row_id"]], rtol=1e-4)
    assert (cells["cell_type"] == np.where(cells["row_id"] % 3 == 0, "B", "T")).all()


def test_a_column_the_input_lacks_is_declined_before_the_analysis(proj, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    code = example().main(["--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--keep-obs", "cell_type"])
    assert code == 2
    assert "cell_type" in capsys.readouterr().err
    assert not (out / "report.json").exists()


def test_the_settings_a_new_user_edits_are_at_the_top() -> None:
    """The example is meant to be dropped in and edited: every setting it
    needs is a plain name near the top, and the defaults run the fixture."""
    mod = example()
    for name in ("INPUT", "OUT", "PROJECT", "RAM_GB", "THREADS", "MODE", "MIN_GENES",
                 "MIN_CELLS", "N_HVG", "N_PCS", "NEIGHBORS", "RESOLUTION", "SEED", "UMAP",
                 "KEEP_OBS", "COLOR_BY", "EXPORT"):
        assert hasattr(mod, name), name
    text = (EXAMPLES / "06_large_run.py").read_text()
    assert text.index("INPUT =") < text.index("import argparse")
    assert mod.INPUT == "" and mod.parse([]).paths == []


def test_pca_uses_only_the_selected_genes(proj, tmp_path) -> None:
    """The highly variable genes are what PCA reads: the gene count drops to
    N_HVG before PCA (``subset=True``)."""
    out = tmp_path / "out"
    code = example().main(["--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--n-hvg", "300", "--no-umap", "--no-export"])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["dataset"]["genes_for_pca"] == 300
    assert "umap" not in [s["step"] for s in report["steps"]]


def test_too_little_disk_for_the_copy_is_declined_before_anything_runs(proj, tmp_path,
                                                                       monkeypatch, capsys):
    import collections
    import shutil

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(10**12, 10**12, 1000))
    out = tmp_path / "out"
    code = example().main(["--out", str(out), "--project", str(tmp_path / "p"),
                           "--ram-gb", "1", "--mode", "out-of-core"])
    assert code == 2
    assert "copy of the counts" in capsys.readouterr().err
    assert not (out / "report.json").exists()


def _shards(tmp_path, edit=None):
    """Two shards of the fixture; ``edit(k, part)`` may change shard k."""
    import anndata as ad

    whole = ad.read_h5ad(EXAMPLES.parent / "tests/data/medium_10kx2k.h5ad")
    folder = tmp_path / "shards"
    folder.mkdir()
    for k, rows in enumerate((slice(0, 5000), slice(5000, 10000))):
        part = whole[rows].copy()
        part.obs = part.obs[[]]
        if edit:
            edit(k, part)
        part.write_h5ad(folder / f"{k}.h5ad")
    return folder


def test_normalized_values_in_x_are_declined_before_the_run(proj, tmp_path, capsys) -> None:
    def normalize(k, part):
        if k == 1:
            part.X = part.X * 0.5
    folder = _shards(tmp_path, normalize)
    out = tmp_path / "out"
    args = [str(folder), "--out", str(out), "--project", str(tmp_path / "p"), "--ram-gb", "1",
            "--min-genes", "100", "--no-umap"]
    assert example().main(args) == 2
    assert "does not hold raw counts" in capsys.readouterr().err
    assert not (out / "report.json").exists()
    assert example().main(args + ["--not-counts"]) == 0


def test_files_that_share_no_gene_names_are_declined(proj, tmp_path, capsys) -> None:
    def rename(k, part):
        if k == 1:
            part.var_names = [f"ENSG{i:011d}" for i in range(part.n_vars)]
    folder = _shards(tmp_path, rename)
    code = example().main([str(folder), "--out", str(tmp_path / "out"), "--project",
                           str(tmp_path / "p"), "--ram-gb", "1", "--min-genes", "100"])
    assert code == 2
    err = capsys.readouterr().err
    assert "share no genes" in err and "names its genes the same way" in err


def test_the_memory_plot_names_every_step_that_can_be_seen() -> None:
    """The 22M run's memory plot named five of ten steps: the narrow ones, and
    the qc step whose name is a whole sentence, had no label."""
    sys.path.insert(0, str(EXAMPLES))
    tracker = load("resources_labels", "_resources.py")
    profile = [("open", 27.1), ("qc (and the one-time copy of the counts)", 2085.9),
               ("normalize", 316.7), ("hvg", 542.7), ("pca", 1537.0), ("neighbors", 1175.8),
               ("leiden", 624.7), ("umap", 3335.5), ("figures", 27.1), ("export", 331.6)]
    steps, t = [], 0.0
    for name, seconds in profile:
        steps.append({"step": name, "start_s": t, "seconds": seconds})
        t += seconds
    labels = {name: (x, turned) for x, name, turned in tracker.step_labels(steps, t)}
    assert set(labels) == {"qc", "normalize", "hvg", "pca", "neighbors", "leiden", "umap", "export"}
    # Slivers (open, figures: 0.3% of the run) are left out; a narrow step is turned.
    assert labels["normalize"][1] and labels["export"][1]
    assert not labels["qc"][1] and not labels["pca"][1] and not labels["umap"][1]
    # Each label sits in the middle of its step.
    assert labels["pca"][0] == pytest.approx(steps[4]["start_s"] + steps[4]["seconds"] / 2)
