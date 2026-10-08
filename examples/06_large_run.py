"""6 — Analyse one large dataset kept as many .h5ad files, end to end.

Edit SETTINGS below for your data, then run:

    python examples/06_large_run.py

Every setting can also be given on the command line (``--help`` lists them),
which is how a Slurm job script, or a second run with other values, passes
them.

It is for one dataset stored as several .h5ad files of the same kind: raw
counts in X, one row per cell, the same genes (or a shared subset), such as
the shards a multi-million-cell collection is distributed in. A folder, a
pattern or a single file all work. With no input it runs on the bundled
fixture, which checks everything, figures included, in seconds before hours
are spent on real data (the fixture is random numbers, so its UMAP is a
featureless blob).

Steps, each timed while memory is sampled twice a second:

    plan       what the run needs in memory and on disk, read from the files'
               description alone; declines if no mode fits
    check      that X holds raw counts (whole, non-negative numbers) and that
               the files share their gene names; declines otherwise, before
               hours go into an analysis that would mean nothing
    qc         per-cell and per-gene metrics; cells with fewer than MIN_GENES
               genes and genes in fewer than MIN_CELLS cells removed. Out of
               core this step includes the one-time copy of the counts.
    normalize  counts scaled to 10,000 per cell, then log1p
    hvg        the N_HVG most variable genes; PCA and every later step use
               only these
    pca, neighbors, leiden, umap
    figures    quality control, variance explained, cluster sizes and UMAPs,
               drawn from every cell at any size (ps.pl.overview)
    export     one row per cell

Written to OUT:

    run_summary.md   what was run, on what machine, the time and memory of
                     each step and the disk used, ready for a methods section
    report.json      the same as numbers
    resources.csv    memory held, memory free and disk free, twice a second;
                     resources.png draws it against time
    cells.tsv.gz     one row per cell kept: name, file (dataset), genes,
                     counts, cluster, UMAP, and the input columns in KEEP_OBS
    qc.png, pca.png, clusters.png, umap.png, and umap_<column>.png for each
    column in COLOR_BY

PROJECT holds Polariseq's working files. Out of core that is a copy of the
counts, written compact and lossless (about one byte per stored value on count
data, an eighth of the plain layout), and, while PCA runs, a copy of the
selected genes; the plan prints the most the folder will hold. Put it on a fast
local or parallel file system. The copy of the counts is reused when the same
unchanged files are run again. Its README.md says what may be deleted.

Cell names often repeat across files (collections number their cells from 0
in each file), so a cell is identified by dataset and cell together, or by an
id column carried with KEEP_OBS. Setting up another machine for this is in
examples/NODE_SETUP.md.
"""
# ruff: noqa: E402  (the settings come first, for whoever edits them)
from __future__ import annotations

# =============================================================================
# SETTINGS: edit these for your data. Paths may be absolute or relative to the
# folder you run from. The command line overrides any of them.
# =============================================================================
INPUT = ""           # a folder of .h5ad files, a pattern such as "data/part_*.h5ad",
                     # or one file; "" runs the bundled fixture
OUT = ""             # figures, tables and logs; "" = examples/output/large_run
PROJECT = ""         # Polariseq's working files; "" = OUT/project
RAM_GB = None        # memory Polariseq may use (GB); None = what is free now
THREADS = None       # worker threads; None = the Slurm allocation if any, else all cores
MODE = "auto"        # "auto" chooses from the plan; or "memory" or "out-of-core"
MIN_GENES = 200      # fewest genes detected for a cell to be kept
MIN_CELLS = 3        # fewest cells a gene must be detected in to be kept
N_HVG = 2000         # highly variable genes: the genes PCA uses
N_PCS = 50           # principal components
NEIGHBORS = 15       # neighbours per cell in the graph
RESOLUTION = 1.0     # Leiden resolution: higher gives more, smaller clusters
SEED = 0             # one seed for PCA, the graph, Leiden and UMAP
UMAP = True          # compute and draw the UMAP (the slowest step at scale)
KEEP_OBS: list[str] = []   # input columns to carry into cells.tsv.gz,
                           # e.g. ["cell_type", "donor_id"]
COLOR_BY: list[str] = []   # also draw the UMAP coloured by these input columns
EXPORT = True        # write cells.tsv.gz
# =============================================================================

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import polariseq as ps
from polariseq.planning import store_bytes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _resources import GB, ResourceLog, describe_machine

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "tests/data/medium_10kx2k.h5ad"


def _names(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*",
                    help="a folder of .h5ad files, a pattern, or files (default: INPUT, "
                         "else the bundled fixture)")
    ap.add_argument("--out", default=OUT or str(HERE / "output/large_run"),
                    help="figures, tables and logs")
    ap.add_argument("--project", default=PROJECT or None,
                    help="Polariseq's working files (default: OUT/project)")
    ap.add_argument("--ram-gb", type=float, default=RAM_GB,
                    help="memory Polariseq may use (default: what is free)")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help="worker threads (default: the Slurm allocation, else all cores)")
    ap.add_argument("--mode", choices=["auto", "memory", "out-of-core"], default=MODE)
    ap.add_argument("--min-genes", type=int, default=None,
                    help=f"fewest genes for a cell to be kept (default {MIN_GENES}; "
                         "100 for the fixture)")
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS,
                    help="fewest cells for a gene to be kept")
    ap.add_argument("--n-hvg", type=int, default=N_HVG, help="highly variable genes, used by PCA")
    ap.add_argument("--workflow", choices=["polariseq", "scanpy"], default="polariseq",
                    help="the defaults of every step (ps.set_workflow): 'polariseq' (genes among "
                         "those detected in more than 1%% of cells, normalization over the "
                         "selected genes, scaled PCA, multilevel UMAP start) or 'scanpy'")
    ap.add_argument("--hvg-flavor", choices=["seurat", "seurat_floor", "scarf"], default=None,
                    help="how genes are selected: 'seurat' (Scanpy's, the default) or 'scarf' "
                         "(Scarf's: variance over a fitted floor, genes detected in more than "
                         "1%% of cells) or 'seurat_floor' (Scanpy's ranking among genes detected "
                         "in more than 1%% of cells)")
    ap.add_argument("--renormalize", action=argparse.BooleanOptionalAction, default=None,
                    help="after the genes are selected, normalize each cell over those genes only "
                         "(to 1,000, then log) and standardize each gene before PCA, as Scarf does "
                         "(default: on in the 'polariseq' workflow)")
    ap.add_argument("--scarf-normalization", dest="renormalize", action="store_const", const=True,
                    help="the same as --renormalize")
    ap.add_argument("--umap-init",
                    choices=["spectral", "pca", "kmeans", "multilevel", "leiden", "random"],
                    default=None,
                    help="where UMAP starts (kmeans: Scarf's start)")
    ap.add_argument("--umap-min-dist", type=float, default=None,
                    help="UMAP's minimum distance (default: the workflow's)")
    ap.add_argument("--umap-epochs", type=int, default=0,
                    help="UMAP rounds (0: 200 above 10,000 cells)")
    ap.add_argument("--n-pcs", type=int, default=N_PCS)
    ap.add_argument("--neighbors", type=int, default=NEIGHBORS)
    ap.add_argument("--resolution", type=float, default=RESOLUTION, help="Leiden resolution")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--umap", action=argparse.BooleanOptionalAction, default=UMAP,
                    help="compute and draw the UMAP")
    ap.add_argument("--keep-obs", default=",".join(KEEP_OBS),
                    help="comma-separated input columns to carry into cells.tsv.gz")
    ap.add_argument("--color-by", default=",".join(COLOR_BY),
                    help="comma-separated input columns to colour extra UMAPs by")
    ap.add_argument("--no-export", action="store_true", default=not EXPORT,
                    help="do not write cells.tsv.gz")
    ap.add_argument("--not-counts", action="store_true",
                    help="analyse X even if it does not look like raw counts")
    args = ap.parse_args(argv)
    if not args.paths and INPUT:
        args.paths = [INPUT]
    return args


def gb(nbytes: float) -> str:
    """Bytes as gigabytes, with the digits that mean something at that size."""
    v = nbytes / GB
    return f"{v:.0f} GB" if v >= 10 else f"{v:.2f} GB" if v >= 0.01 else f"{v * 1000:.1f} MB"


def duration(seconds: float) -> str:
    return (f"{seconds:.1f} s" if seconds < 90 else f"{seconds / 60:.1f} min"
            if seconds < 5400 else f"{seconds / 3600:.2f} h")


def resolve_inputs(paths: list[str]) -> list[str]:
    """The .h5ad files named: a file stands for itself, a folder for its
    .h5ad files in name order, a pattern (the shell may not expand it, as on
    Windows) for the files it matches."""
    import glob
    import re

    def natural(name: str) -> list:   # 2.h5ad before 10.h5ad
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

    files: list[str] = []
    for p in paths:
        if Path(p).is_dir():
            found = sorted((str(f) for f in Path(p).glob("*.h5ad")), key=natural)
            if not found:
                raise SystemExit(f"no .h5ad files in {p}")
            files += found
        elif any(c in p for c in "*?["):
            files += sorted(glob.glob(p), key=natural)
        else:
            files.append(p)
    missing = [f for f in files if not Path(f).is_file()]
    if not files or missing:
        raise SystemExit(f"no such .h5ad file: {missing[0] if missing else paths}")
    return files


def show_plans(path: str | list[str], args: argparse.Namespace) -> dict:
    """What the run would cost, from the files' description alone."""
    plans = {}
    for storage in ("ram", "spill"):
        p = ps.plan(path, n_hvg=args.n_hvg, n_pcs=args.n_pcs, storage=storage,
                    with_umap=args.umap, verbose=False)
        plans[storage] = p
        verdict = "fits" if p.fits else f"does NOT fit (needs a budget of {p.min_budget_gb:.1f} GB)"
        where = "in memory" if storage == "ram" else "out of core"
        print(f"  {where:12s} peak {p.peak_gb:7.2f} GB, limited by '{p.limiting_step}', "
              f"{p.rows_per_block:,} cells per block: {verdict}")
    p = plans["spill"]
    print(f"  {p.n_cells:,} cells, {p.n_genes:,} genes, {p.nnz:,} stored values")
    print(f"  project folder out of core: at most {gb(p.disk_peak_bytes)} (copy of the counts "
          f"{gb(p.disk['copy'])}, PCA's copy of the selected genes about "
          f"{gb(p.disk['pca_copy'])}, results {gb(p.disk['results'])})\n")
    return plans


def check_counts(path: str, n: int = 200_000) -> str | None:
    """Why X of this file does not look like raw counts, or None. Reads the
    first ``n`` stored values (h5py, which anndata installs; skipped without
    it). Normalized or log-transformed values would run and mean nothing."""
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(path, "r") as f:
        x = f.get("X")
        if x is None:
            return None
        if isinstance(x, h5py.Group):
            vals = x["data"][:n] if "data" in x else np.zeros(0)
        else:
            vals = x[: max(1, n // max(1, x.shape[1]))].ravel()[:n]
        vals = np.asarray(vals, dtype=np.float64)
        wrong = vals[(vals != np.floor(vals)) | (vals < 0)]
        if wrong.size == 0:
            return None
        hint = next((f" The file also has {w}; Polariseq reads X, so write the raw counts to "
                     "X first." for w in ("layers/counts", "raw/X") if w in f), "")
    return (f"X of {Path(path).name} does not hold raw counts: {wrong.size:,} of the first "
            f"{vals.size:,} values are not whole non-negative numbers (for example "
            f"{wrong[0]:g}).{hint}")


def check_genes(plan) -> tuple[bool, str | None]:
    """Whether several files can be analysed as one, and what to say about
    the genes they share. Files whose gene names are of different kinds
    (symbols in one, Ensembl ids in another) share almost none."""
    ds = plan.datasets
    if len(ds) < 2:
        return True, None
    kept = min(d["genes_kept"] for d in ds)
    smallest = min(d["n_genes"] for d in ds)
    if kept == 0:
        return False, ("the files share no gene names. Check that every file names its genes "
                       "the same way (all symbols, or all Ensembl ids).")
    if kept < 0.5 * smallest:
        return True, (f"WARNING: only {kept:,} genes are in every file, against {smallest:,} in "
                      "the smallest. Check that every file names its genes the same way.")
    dropped = [d for d in ds if d["genes_kept"] < d["n_genes"]]
    if dropped:
        return True, (f"note: {len(dropped)} file(s) have genes the others lack; only the "
                      f"{kept:,} genes in every file are kept.")
    return True, None


def disk_problem(project: Path, need: float, what: str) -> str | None:
    """Why the disk under ``project`` cannot take ``need`` bytes more, or None."""
    free = shutil.disk_usage(project).free
    if free < 1.05 * need:
        return f"{what} needs about {gb(need)} in {project}, which has {gb(free)} free"
    return None


def export_cells(adata, out: Path, keep: tuple[str, ...] = (), chunk: int = 2_000_000) -> Path:
    """One row per cell kept: name, the file it came from (when several were
    read as one), qc numbers, cluster, the input columns asked for and, if
    computed, the UMAP, written in chunks so that a hundred million cells need
    no copy of the table in memory. Names can repeat across files; a cell is
    identified by dataset and cell, or by an id column kept with KEEP_OBS.
    The results otherwise live in the project folder as memory-mapped files;
    this is the copy that outlives it."""
    import pandas as pd

    names = adata.obs_names
    cols = {"dataset": adata.obs["dataset"]} if "dataset" in adata.obs else {}
    cols.update({k: np.asarray(adata.obs[k]) for k in ("n_genes", "n_counts", "leiden")})
    cols.update({k: adata.obs[k] for k in keep})   # labels stay compact categoricals
    umap = np.asarray(adata.obsm["X_umap"]) if "X_umap" in adata.obsm else None
    path = out / "cells.tsv.gz"
    for i in range(0, adata.n_obs, chunk):
        j = min(i + chunk, adata.n_obs)
        frame = pd.DataFrame({"cell": names[i:j], **{k: v[i:j] for k, v in cols.items()}})
        if umap is not None:
            frame["umap1"], frame["umap2"] = umap[i:j, 0], umap[i:j, 1]
        frame.to_csv(path, sep="\t", index=False, header=i == 0, mode="w" if i == 0 else "a",
                     compression="gzip")
    return path


def _genes_text(s: dict) -> str:
    default = "seurat_floor" if s.get("workflow") == "polariseq" else "seurat"
    flavor = s.get("hvg_flavor") or default
    return {"seurat": "Scanpy's seurat ranking",
            "seurat_floor": "Scanpy's seurat ranking among genes detected in more than 1% of cells",
            "scarf": "Scarf's floor-trend method"}[flavor]


def _norm_text(s: dict) -> str:
    if s.get("renormalize"):
        return ("each cell then normalized again over the selected genes (to 1,000, log) and each "
                "gene standardized before PCA")
    return "no second normalization"


def _umap_text(s: dict) -> str:
    if not s.get("umap"):
        return "no UMAP"
    pol = s.get("workflow") == "polariseq"
    init = s.get("umap_init") or ("multilevel" if pol else "spectral")
    md = s.get("umap_min_dist") or (0.5 if pol else 0.1)
    ep = s.get("umap_epochs") or "the workflow's"
    return f"UMAP from the {init} start, minimum distance {md}, {ep} rounds"


def write_summary(out: Path, r: dict) -> Path:
    """The run in prose and one table, for a methods section or a lab notebook."""
    d, m, b = r["dataset"], r["machine"], r["budget"]
    s = r["settings"]
    lines = [
        "# Run summary", "",
        f"Polariseq {m['polariseq'].get('python_version', '?')}, "
        f"{time.strftime('%Y-%m-%d %H:%M')}, mode: {r['mode']}.", "",
        "## Data", "",
        f"- {d['cells_in_file']:,} cells, {d['genes']:,} genes and {d['stored_values']:,} "
        f"stored values in {len(d['files'])} file(s)",
        f"- after quality control: {d['cells_after_qc']:,} cells and {d['genes_after_qc']:,} "
        f"genes; {d['genes_for_pca']:,} highly variable genes used for PCA; "
        f"{d['clusters']} clusters", "",
        "## Settings", "",
        f"- cells kept with at least {s['min_genes']} genes; genes kept in at least "
        f"{s['min_cells']} cells; counts scaled to 10,000 per cell, then log1p",
        f"- workflow '{s.get('workflow', 'scanpy')}': {s['n_hvg']} highly variable genes "
        f"({_genes_text(s)}); {_norm_text(s)}; {s['n_pcs']} principal components, "
        f"{s['neighbors']} neighbours, Leiden resolution {s['resolution']}; "
        f"{_umap_text(s)}; seed {s['seed']}", "",
        "## Machine", "",
        f"- processor: {m.get('cpu') or 'unknown'}",
        f"- cores: {m.get('cores_physical', '?')} physical, {m.get('cores_logical', '?')} "
        f"logical; the run used {b['threads']} threads",
        f"- memory: {m.get('memory_gb', '?')} GB; Polariseq's budget {b['ram_gb']:.0f} GB",
    ]
    if "scheduler_allocation" in m:
        a = m["scheduler_allocation"]
        lines.append(f"- scheduler allocation: {a.get('cpus', '?')} cores"
                     + (f", {int(a['memory_mb']) / 1000:.0f} GB" if "memory_mb" in a else ""))
    lines += [
        f"- project storage: {m.get('project_filesystem', 'unknown file system')}, "
        f"{m.get('project_disk_free_gb', '?')} GB free at the start",
        f"- operating system: {m['os']}; Python {m['python']}", "",
        "## Time and memory", "",
        "| step | time | peak memory held (GB) |", "|---|---:|---:|",
    ]
    lines += [f"| {st['step']} | {duration(st['seconds'])} | {st['peak_rss_gb']:.1f} |"
              for st in r["steps"]]
    pk = r["peak_memory_gb"]
    lines += [f"| **total** | **{duration(r['seconds_total'])}** | **{pk['held_os']:.1f}** |", "",
              f"Lowest free memory on the machine during the run: "
              f"{pk['lowest_free']:.1f} GB." if pk["lowest_free"] is not None else "",
              f"Project folder at the end {r['disk_gb']['project']:.1f} GB; "
              + (f"written during the run {r['disk_gb']['written_during_run']:.1f} GB."
                 if r["disk_gb"]["written_during_run"] is not None else ""), "",
              "## Outputs", "", ", ".join(sorted(r["outputs"]))]
    path = out / "run_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    files = resolve_inputs(args.paths) if args.paths else [str(FIXTURE)]
    path = files if len(files) > 1 else files[0]    # a list reads as one dataset
    if args.min_genes is None:   # a property of the data: the fixture has only 2,000 genes
        args.min_genes = MIN_GENES if args.paths else min(MIN_GENES, 100)
    keep = tuple(_names(args.keep_obs))
    color_by = _names(args.color_by)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None
    ps.set_budget(ram_gb=args.ram_gb, threads=threads)
    ps.set_workflow(args.workflow)
    if args.renormalize is None:   # on in Polariseq's workflow, off in Scanpy's
        args.renormalize = args.workflow == "polariseq"
    project = ps.project(args.project or out / "project")
    budget = ps.budget()
    shown = path if isinstance(path, str) else f"{files[0]} and {len(files) - 1} more files"
    print(f"input    {shown}\nproject  {project.path}\n"
          f"budget   {budget.ram_bytes / GB:.1f} GB of memory, {budget.threads} threads\n\n"
          "plan, before anything runs:")
    try:
        plans = show_plans(path, args)
    except ValueError as e:   # the planner refuses files it cannot read as one dataset
        print(f"declined: {e}. Check that every file names its genes the same way (all "
              "symbols, or all Ensembl ids).", file=sys.stderr)
        return 2
    allowed = {"auto": ("ram", "spill"), "memory": ("ram",), "out-of-core": ("spill",)}[args.mode]
    if not any(plans[s].fits for s in allowed):
        print("declined: no mode fits this budget. Raise --ram-gb, free memory, or use a "
              "machine with more.", file=sys.stderr)
        return 2
    wrong = [w for w in {check_counts(files[0]), check_counts(files[-1])} if w]
    if wrong and not args.not_counts:
        print(f"declined: {wrong[0]} Run with --not-counts to analyse it anyway.", file=sys.stderr)
        return 2
    for w in wrong:
        print(f"WARNING: {w}", file=sys.stderr)
    ok, note = check_genes(plans["spill"])
    if not ok:
        print(f"declined: {note}", file=sys.stderr)
        return 2
    if note:
        print(note + "\n", flush=True)
    spill = args.mode == "out-of-core" or (args.mode == "auto" and not plans["ram"].fits)
    copied = any((project.path / "runs").glob("*/matrix/source.json"))
    problem = spill and not copied and disk_problem(
        project.path, plans["spill"].disk["copy"], "the copy of the counts")
    if problem:
        print(f"declined: {problem}. Point --project at a disk with more room.", file=sys.stderr)
        return 2

    t_wall = time.perf_counter()
    with ResourceLog(watch=project.path, verbose=True) as log:
        with log.step("open"):
            modes = {} if args.mode == "auto" else {"mode": args.mode}
            adata = ps.read_h5ad(path, **modes)
        print(f"mode: {adata.mode}\n", flush=True)
        absent = [k for k in (*keep, *color_by) if k not in adata.obs]
        if absent:   # before the long steps, not after
            print(f"declined: KEEP_OBS/COLOR_BY name columns the input does not have: "
                  f"{', '.join(absent)}. It has: {', '.join(adata.obs)}", file=sys.stderr)
            return 2
        with log.step("qc (and the one-time copy of the counts)"):
            ps.pp.calculate_qc_metrics(adata)
            # Every cell's metrics, before the filter, so the quality-control
            # figure can show the cells it removed.
            qc_before = {k: np.asarray(adata.obs[k], dtype=np.float32)
                         for k in ("n_genes", "n_counts")}
            ps.pp.filter_cells(adata, min_genes=args.min_genes)
            ps.pp.filter_genes(adata, min_cells=args.min_cells)
        genes_after_qc = int(adata.n_vars)
        kept = adata.n_obs / plans["spill"].n_cells
        if kept < 0.5:   # a threshold that does not fit the data: say so before hours are spent
            print(f"\nWARNING: MIN_GENES {args.min_genes} removed {1 - kept:.0%} of the cells. "
                  "Look at the genes per cell of your data (qc.png) and lower it.\n",
                  file=sys.stderr, flush=True)
        with log.step("normalize"):
            ps.pp.normalize_total(adata, target_sum=1e4)
            ps.pp.log1p(adata)
        with log.step("hvg"):
            if args.hvg_flavor is None:   # the workflow's selection
                ps.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, subset=True)
            elif args.hvg_flavor == "seurat_floor":
                ps.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat",
                                            min_cells=int(0.01 * adata.n_obs), subset=True)
            else:
                ps.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor=args.hvg_flavor,
                                            subset=True)
        if adata.mode == "out-of-core":   # PCA copies the selected genes: is there room?
            values = int(np.sum(np.asarray(adata.var["n_cells"], dtype=float)))
            need = store_bytes(values, adata.n_obs)   # in the layout the copy is written in
            print(f"PCA's copy of the selected genes: {gb(need)}", flush=True)
            problem = disk_problem(project.path, need, "PCA's copy of the selected genes")
            if problem:
                print(f"stopped before PCA: {problem}. Free space, or point --project at a "
                      "disk with more room; the copy of the counts will be reused.",
                      file=sys.stderr)
                return 2
        if args.renormalize:
            with log.step("renormalize"):
                ps.pp.renormalize(adata, target_sum=1000)
        with log.step("pca"):
            ps.pp.pca(adata, n_components=args.n_pcs, seed=args.seed, scale=bool(args.renormalize))
        with log.step("neighbors"):
            ps.pp.neighbors(adata, n_neighbors=args.neighbors, seed=args.seed)
        with log.step("leiden"):
            ps.tl.leiden(adata, resolution=args.resolution, seed=args.seed)
        if args.umap:
            with log.step("umap"):
                ps.tl.umap(adata, seed=args.seed, init=args.umap_init, min_dist=args.umap_min_dist,
                           n_epochs=args.umap_epochs)
        with log.step("figures"):
            figures = ps.pl.overview(adata, str(out), color_by=color_by,
                                     thresholds={"n_genes": args.min_genes},
                                     qc_before=qc_before)
        n_clusters = int(len(np.unique(np.asarray(adata.obs["leiden"]))))
        if not args.no_export:
            with log.step("export"):
                export_cells(adata, out, keep)
    total = time.perf_counter() - t_wall

    logs = log.save(out)
    status = project.status()
    use = log.summary()
    machine = describe_machine(project.path)
    machine.update(polariseq=ps.get_build_info())
    report = {
        "dataset": {"files": files, "cells_in_file": plans["spill"].n_cells,
                    "genes": plans["spill"].n_genes, "stored_values": plans["spill"].nnz,
                    "cells_after_qc": int(adata.n_obs), "genes_after_qc": genes_after_qc,
                    "genes_for_pca": int(adata.n_vars), "clusters": n_clusters},
        "settings": {k: v for k, v in vars(args).items() if k not in ("paths", "out", "project")},
        "mode": adata.mode,
        "budget": {"ram_gb": budget.ram_bytes / GB, "threads": budget.threads},
        "plan": {s: {"fits": p.fits, "peak_gb": p.peak_gb, "limiting_step": p.limiting_step,
                     "cells_per_block": p.rows_per_block} for s, p in plans.items()},
        "seconds_total": total,
        "peak_memory_gb": {"held_sampled": use["peak_rss_gb_sampled"],
                           "held_os": use["peak_rss_gb_os"],
                           "lowest_free": use["min_free_ram_gb"]},
        "disk_gb": {"project": sum(status.values()) / GB,
                    "written_during_run": use["disk_written_gb"],
                    "by_part": {k: v / GB for k, v in status.items()}},
        "steps": use["steps"],
        "machine": machine,
        "outputs": [Path(f).name for f in figures.values()]
                   + [logs["csv"].name, logs["png"].name, "report.json", "run_summary.md"]
                   + ([] if args.no_export else ["cells.tsv.gz"]),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    write_summary(out, report)

    print("\nstep".ljust(46), "seconds".rjust(9), "peak memory (GB)".rjust(18))
    for s in use["steps"]:
        print(f"  {s['step']:42s} {s['seconds']:9.1f} {s['peak_rss_gb']:18.2f}")
    print(f"\n{plans['spill'].n_cells:,} cells in the file, {adata.n_obs:,} after quality control, "
          f"{n_clusters} clusters; {total / 60:.1f} min; peak memory held "
          f"{use['peak_rss_gb_os']:.2f} GB; project {gb(sum(status.values()))} on disk")
    print(f"wrote {out}/: run_summary.md, report.json, {logs['csv'].name}, "
          f"{logs['png'].name} and the figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
