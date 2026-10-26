#!/usr/bin/env python3
"""Independent validation of Polariseq's published claims.

Run this yourself. It is the whole point of the file: every performance
number Polariseq publishes should be reproducible by someone who does not
work on it, on hardware we have never seen, without taking our word for
anything.

    python benchmarks/validate.py --trials 5

What it does
------------
Runs the same fixed pipeline — load, QC, filter, normalise, log1p, HVG, PCA,
neighbours, Leiden (and optionally UMAP) — through four implementations:

    polariseq-ram      ours, in memory
    polariseq-spill    ours, out-of-core (opt in with --arms)
    scanpy-default     scanpy exactly as shipped
    scanpy-tuned       scanpy configured by someone who wants it to win

across several dataset sizes, `--trials` times each, and reports every
number as **mean ± half-range (n)** rather than as a single figure.

Why it is built the way it is
-----------------------------
Each of these is a control against a specific way this kind of comparison
goes wrong.

- **One fresh process per (dataset, arm, trial).** Peak memory is a
  process-wide high-water mark, so two arms in one interpreter each inherit
  the other's worst moment. This is what "no run affects the other" requires.
- **Both libraries get the same thread count**, exported into all eight
  thread environment variables. scanpy defaults to `n_jobs=1`, which pins
  numba inside pynndescent; comparing at different thread counts measures
  the launcher, not the libraries.
- **The memory budget is pinned.** Otherwise Polariseq's block size floats
  with ambient free RAM and two trials on one machine disagree with each
  other.
- **The page cache is warmed identically before every run, and the warm read
  is timed**, so "was one arm unfairly pre-cached?" is a number on the
  record rather than a worry.
- **Arm order rotates every trial**, so no arm always runs first on the
  quietest machine.
- **The answers are compared before the speeds.** A speed comparison
  between implementations computing different things is not a speed
  comparison. PCA by principal angles and variance capture, clustering by
  ARI against the reference's own seed-to-seed variability, HVG by Jaccard.
- **Runs that prove nothing are excluded and named**, never dropped
  silently: anything that swapped, crashed, or whose two independent memory
  samplers disagreed.

Data
----
By default it downloads **real, public 10x Genomics datasets** — no login,
no licence click, stable URLs, and the same files the field already
benchmarks on. Four of them, 2.7k to 20k cells, across two tissues so a
result is not an artefact of one cell type. They are cached, so a re-run
does not re-download.

Real data is the default because synthetic data cannot validate a quality
claim: on generated blobs every arm reaches ARI 1.0, since well-separated
types are trivial to cluster. It also has the wrong shape — real dropout,
the long tail of rarely-detected genes and ambient background are what the
HVG and neighbour steps actually have to cope with.

`--synthetic` generates data locally instead, for a machine with no network.
The report always states which was used.

Exit status is 0 if every arm completed and no run was blocked, 1 otherwise,
so this can be wired into CI.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from telemetry.analyze import (  # noqa: E402
    MIN_TRIALS_FOR_STATS,
    quality,
    stat,
    summarize,
)
from telemetry.arms import ARMS, CANONICAL_STEPS, dataset_facts  # noqa: E402
from telemetry.datasets import (  # noqa: E402
    DEFAULT_LADDER,
    PUBLIC_DATASETS,
    fetch_all,
)
from telemetry.equivalence import agreement_floors, compare_all  # noqa: E402
from telemetry.record import Params, load_records  # noqa: E402
from telemetry.runner import run_one  # noqa: E402

#: Synthetic sizes, chosen to span the range where behaviour changes: small
#: enough to finish on a laptop, large enough that the out-of-core path and
#: the graph stages start to matter.
#:
#: Gene counts are realistic rather than convenient. Cost scales with
#: nonzeros, and nonzeros are `cells x genes x density` — so a 12,000-gene
#: matrix at 4 % density gives ~480 nonzeros per cell where real droplet
#: data gives 850-1,100 (measured: pbmc3k 846, heart_10k 1,114). Shrinking
#: the gene axis to save disk would quietly make the benchmark easier than
#: the workload it stands in for.
DEFAULT_SIZES = ((3_000, 20_000), (20_000, 25_000), (60_000, 30_000))

DEFAULT_ARMS = ("polariseq-ram", "scanpy-default", "scanpy-tuned")


#: Target fraction of nonzero entries. Real droplet scRNA-seq sits here —
#: measured on the corpus this project benchmarks against: pbmc3k 2.6 %,
#: heart_10k 3.7 %. Density is not cosmetic: every cost in both libraries
#: scales with nonzeros, so a generator that produces 34 % density (as a
#: naive Poisson does) makes a 20k x 12k matrix 82 M nonzeros instead of
#: 8 M, which both misrepresents the workload and fills a laptop's disk.
SYNTHETIC_DENSITY = 0.04


def make_synthetic(path: Path, n_cells: int, n_genes: int, seed: int = 0) -> None:
    """Write a synthetic sparse counts `.h5ad` with real cluster structure.

    Poisson counts over a handful of cell types with distinct marker
    programmes, then thinned to :data:`SYNTHETIC_DENSITY` by the dropout
    that makes real droplet data sparse.

    Two properties matter and neither is decorative:

    - **Structure.** Unstructured noise makes approximate neighbour search
      behave pathologically — pynndescent takes 11 s on a 10k noise matrix —
      which would misrepresent both libraries rather than either one.
    - **Sparsity.** Both libraries' costs scale with nonzeros, not with
      cells x genes, so getting density wrong changes what is being measured.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_types = 8
    labels = rng.integers(0, n_types, n_cells)

    # Per-type expression programme: a shared background plus type-specific
    # markers, so the clusters are separable but not trivially so.
    base = rng.gamma(0.35, 2.0, n_genes).astype(np.float32)
    programmes = np.tile(base, (n_types, 1))
    for t in range(n_types):
        markers = rng.choice(n_genes, size=max(10, n_genes // 40), replace=False)
        programmes[t, markers] *= rng.uniform(6.0, 25.0, markers.size)

    # Per-gene detection probability, Zipf-ish: a few genes are seen in most
    # cells and the long tail is nearly always dropped, which is what makes
    # real data sparse and what the HVG step has to find signal in.
    detect = rng.beta(0.35, 3.0, n_genes).astype(np.float32)
    detect *= SYNTHETIC_DENSITY / detect.mean()
    np.clip(detect, 0.0, 1.0, out=detect)

    rows, cols, vals = [], [], []
    chunk = 4096  # bounded memory while generating
    for lo in range(0, n_cells, chunk):
        hi = min(lo + chunk, n_cells)
        mu = programmes[labels[lo:hi]] * rng.uniform(0.6, 1.6, (hi - lo, 1))
        keep = rng.random((hi - lo, n_genes)) < detect
        counts = np.where(keep, rng.poisson(mu) + 1, 0).astype(np.float32)
        r, c = np.nonzero(counts)
        rows.append((r + lo).astype(np.int64))
        cols.append(c.astype(np.int64))
        vals.append(counts[r, c])
    x = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cells, n_genes),
        dtype=np.float32,
    )
    adata = ad.AnnData(X=x)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["true_type"] = [f"type_{t}" for t in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--trials", type=int, default=MIN_TRIALS_FOR_STATS,
                   help=f"runs per (dataset, arm); >= {MIN_TRIALS_FOR_STATS} for "
                        f"a number worth quoting (default: %(default)s)")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help=f"comma-separated; known: {','.join(sorted(ARMS))}")
    p.add_argument("--datasets", default="",
                   help="comma-separated .h5ad paths to use instead of the "
                        "public ladder")
    p.add_argument("--public", default=",".join(DEFAULT_LADDER),
                   help=f"public datasets to download; known: "
                        f"{','.join(sorted(PUBLIC_DATASETS))} "
                        f"(default: %(default)s)")
    p.add_argument("--synthetic", action="store_true",
                   help="generate data locally instead of downloading — for "
                        "an offline machine. Cannot validate quality claims: "
                        "every arm scores ARI 1.0 on generated blobs.")
    p.add_argument("--sizes", default="",
                   help="synthetic sizes as cells x genes, e.g. 3000x20000")
    p.add_argument("--threads", type=int, default=4,
                   help="thread count given to *every* arm (default: %(default)s)")
    p.add_argument("--ram-gb", type=float, default=4.0,
                   help="pinned memory budget for Polariseq (default: %(default)s)")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--with-umap", action="store_true",
                   help="include UMAP (slow; off by default)")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--out", default=str(HERE / "results" / "validation"),
                   help="where records and the report are written")
    p.add_argument("--data-dir", default=str(HERE / "results" / "validation_data"))
    p.add_argument("--figures", action="store_true")
    p.add_argument("--clean-data", action="store_true",
                   help="delete generated synthetic files afterwards; by "
                        "default they are kept so re-runs skip regeneration")
    return p


def _sizes(spec: str) -> tuple[tuple[int, int], ...]:
    if not spec:
        return DEFAULT_SIZES
    out = []
    for part in spec.split(","):
        cells, _, genes = part.strip().lower().partition("x")
        out.append((int(cells), int(genes)))
    return tuple(out)


def _fmt(st: dict | None, unit: str = "", places: int = 2) -> str:
    """`mean ± half-range (n)` from a serialised Stat, or an em dash."""
    if not st:
        return "—"
    half = (st["max"] - st["min"]) / 2.0
    mark = "*" if st["underpowered"] else ""
    return f"{st['mean']:.{places}f} ± {half:.{places}f}{unit} (n={st['n']}{mark})"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    results = out / "records"

    print("Polariseq independent validation")
    print(f"  {platform.platform()}")
    print(f"  python {platform.python_version()}")
    print(f"  trials={args.trials}  threads={args.threads}  ram_gb={args.ram_gb}")
    if args.trials < MIN_TRIALS_FOR_STATS:
        print(f"  NOTE: fewer than {MIN_TRIALS_FOR_STATS} trials — every number "
              f"below will be marked `*` as under-powered.")

    # --- inputs -----------------------------------------------------------
    targets: list[tuple[str, str]] = []
    data_dir = Path(args.data_dir)
    if args.datasets:
        data_source = "user-supplied .h5ad files"
        for spec in args.datasets.split(","):
            p = Path(spec.strip())
            if not p.exists():
                print(f"  [skip] {p}: not found", file=sys.stderr)
                continue
            targets.append((p.stem, str(p)))
    elif args.synthetic:
        data_source = "synthetic (generated locally — timing/memory only)"
        print(f"\nGenerating synthetic datasets in {data_dir}")
        print("  NOTE: synthetic data cannot validate clustering quality — "
              "every arm scores ARI 1.0 on generated blobs.")
        for n_cells, n_genes in _sizes(args.sizes):
            label = f"synth_{n_cells // 1000}k"
            path = data_dir / f"{label}.h5ad"
            if not path.exists():
                print(f"  building {label}: {n_cells:,} x {n_genes:,} ...",
                      end="", flush=True)
                make_synthetic(path, n_cells, n_genes)
                print(f" {path.stat().st_size / 1e6:.0f} MB")
            targets.append((label, str(path)))
    else:
        names = tuple(n.strip() for n in args.public.split(",") if n.strip())
        data_source = f"public 10x datasets ({', '.join(names)})"
        print(f"\nFetching real public datasets into {data_dir} "
              f"(cached; re-runs do not re-download)")
        targets = fetch_all(names, data_dir)
    if not targets:
        print("no datasets to run", file=sys.stderr)
        return 1

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"unknown arms: {unknown}; known: {sorted(ARMS)}", file=sys.stderr)
        return 2

    params = Params(
        n_hvg=args.n_hvg, n_pcs=args.n_pcs, with_umap=args.with_umap,
        threads=args.threads, ram_gb=args.ram_gb,
    )

    # --- run --------------------------------------------------------------
    total = len(targets) * len(arms) * args.trials
    done = 0
    for label, path in targets:
        facts = dataset_facts(path)
        print(f"\n=== {label}: {facts['n_cells']:,} x {facts['n_genes']:,}, "
              f"{facts['nnz']:,} nnz ===")
        for trial in range(1, args.trials + 1):
            # Rotate so no arm is always first on the freshest cache.
            k = (trial - 1) % len(arms)
            for arm in arms[k:] + arms[:k]:
                done += 1
                print(f"[{done}/{total}] {label} / {arm} trial{trial} ...",
                      end="", flush=True)
                rec = run_one(
                    arm=arm, path=path, label=label, params=params,
                    results_dir=results, rep=trial, timeout_s=args.timeout,
                    facts=facts,
                )
                print(f" {rec.status} {rec.wall_s:.1f}s "
                      f"uss={rec.peaks['child_uss'] / 1e9:.2f}GB"
                      + (f" — {rec.note[:100]}" if rec.note else ""))

    # --- report -----------------------------------------------------------
    records = load_records(results)
    rows = summarize(records)
    blocked = [r for r in records if not quality(r)["ok"]]

    lines = [
        "# Polariseq validation report",
        "",
        f"- machine: `{platform.platform()}`",
        f"- cpu: `{records[0]['env']['cpu']}` "
        f"({records[0]['env']['total_ram_bytes'] / 1e9:.1f} GB RAM)"
        if records else "",
        f"- polariseq: `{records[0]['env']['git_sha']}`, "
        f"scanpy `{records[0]['env']['versions'].get('scanpy', '?')}`"
        if records else "",
        f"- trials per arm: {args.trials}, threads: {args.threads}, "
        f"budget: {args.ram_gb} GB",
        f"- data: {data_source}",
        "",
        "All figures are **mean ± half-range (n)**. `*` marks fewer than "
        f"{MIN_TRIALS_FOR_STATS} trials.",
        "",
        "## Wall time and memory",
        "",
        "| dataset | arm | pipeline (s) | peak RSS (GB) | peak USS (GB) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        st = r.get("stats") or {}
        lines.append(
            f"| {r['dataset']} | {r['arm_label']} | {_fmt(st.get('pipeline_s'), 's')} "
            f"| {_fmt(st.get('peak_rss_gb'), ' GB')} "
            f"| {_fmt(st.get('peak_uss_gb'), ' GB')} |"
        )

    # Speedups carry their own uncertainty: the ratio of two noisy numbers is
    # noisier than either, so the range is propagated rather than dropped.
    lines += ["", "## Speedup, with the spread propagated", "",
              "| dataset | vs scanpy-default | vs scanpy-tuned |", "|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], {})[r["arm"]] = (r.get("stats") or {}).get("pipeline_s")
    for dataset, arms_ in sorted(by.items()):
        mine = arms_.get("polariseq-ram")
        if not mine or not mine["mean"]:
            continue
        cells = []
        for other in ("scanpy-default", "scanpy-tuned"):
            o = arms_.get(other)
            if not o or not o["mean"]:
                cells.append("—")
                continue
            # Worst/best case ratio from the observed ranges.
            lo = o["min"] / mine["max"] if mine["max"] else float("nan")
            hi = o["max"] / mine["min"] if mine["min"] else float("nan")
            cells.append(f"{o['mean'] / mine['mean']:.2f}x ({lo:.2f}–{hi:.2f})")
        lines.append(f"| {dataset} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Do the arms compute the same answer?", ""]
    floors = agreement_floors(records)
    if floors:
        lines.append("Seed-to-seed floors (the reference against itself):")
        for label, f in sorted(floors.items()):
            lines.append("- " + label + ": "
                         + ", ".join(f"{k} {v:.3f}" for k, v in sorted(f.items())))
        lines.append("")
    comparisons = compare_all(records, n_neighbors=15)
    if not comparisons:
        lines.append("_No `scanpy-tuned` reference in this run, so no comparison._")
    seen = set()
    for c in comparisons:
        key = (c["dataset"], c["arm"])
        if key in seen:
            continue
        seen.add(key)
        ch = c["checks"]
        lines.append(
            f"- **{c['dataset']}** {c['arm']}: **{c['verdict']}** — "
            f"PCA variance ratio {ch.get('pca_variance_ratio')}, "
            f"angle {ch.get('pca_max_angle_deg_at_k_identified')}°, "
            f"HVG Jaccard {ch.get('hvg_jaccard')}, ARI {ch.get('cluster_ari')}"
        )
        if c.get("confounded_by"):
            lines.append(f"  - {c['confounded_by']}")

    lines += ["", "## Runs excluded", ""]
    if blocked:
        for r in blocked:
            why = "; ".join(quality(r)["blocking"])
            lines.append(f"- `{r['dataset']['label']}` / `{r['arm']}`: {why}")
    else:
        lines.append("None — every run completed and none was blocked.")

    # Reproducibility of the *measurement itself*, which a reader should see
    # before trusting any single figure above.
    spreads = [
        ((r.get("stats") or {}).get("pipeline_s") or {}).get("max", 0)
        - ((r.get("stats") or {}).get("pipeline_s") or {}).get("min", 0)
        for r in rows
    ]
    means = [((r.get("stats") or {}).get("pipeline_s") or {}).get("mean", 0) for r in rows]
    rel = [s / m for s, m in zip(spreads, means, strict=True) if m]
    if rel:
        worst = stat(rel)
        lines += ["", "## How reproducible was this machine?", "",
                  f"Relative spread of pipeline time across trials: "
                  f"median {worst.median:.1%}, worst {worst.hi:.1%}. "
                  f"A number quoted without a spread this size is a number "
                  f"quoted without its error bar.", ""]
        # Name the step responsible, because "the benchmark is noisy" is not
        # actionable and "one step is noisy" usually is. Measured here: the
        # whole of a 54% pipeline spread came from `pca` (1.28-3.80 s) while
        # every other step held to within 0.13 s.
        noisiest: list[tuple[float, str, str, str]] = []
        for r in rows:
            for name, st_ in ((r.get("stats") or {}).get("steps") or {}).items():
                if st_ and st_["mean"] > 0.05 and st_["n"] > 1:
                    noisiest.append((
                        (st_["max"] - st_["min"]) / st_["mean"],
                        r["dataset"], r["arm"], name,
                    ))
        noisiest.sort(reverse=True)
        if noisiest:
            lines.append("Steps contributing most of that spread:")
            for relspread, dataset, arm, name in noisiest[:5]:
                lines.append(f"- `{dataset}` / `{arm}` / **{name}**: "
                             f"{relspread:.0%} of its own mean")

    out.mkdir(parents=True, exist_ok=True)
    report = out / "REPORT.md"
    report.write_text("\n".join(x for x in lines if x is not None))
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n" + "\n".join(x for x in lines if x is not None))
    print(f"\nwrote {report}")

    if args.figures:
        from run_telemetry import collect_predictions
        from telemetry.figures import make_panels

        # Panel D is the strongest single claim available — a memory
        # predictor that brackets measurement across a decade of dataset
        # sizes — so the prediction line has to be on it. Built from each
        # record's own pinned budget, never the ambient one.
        predictions = collect_predictions(records, with_umap=args.with_umap)
        for label in sorted({r["dataset"]["label"] for r in records}):
            p = make_panels(records, rows, label,
                            out / "figures" / f"validation_{label}.png",
                            CANONICAL_STEPS, predictions=predictions)
            print(f"wrote {p}")

    if args.synthetic and not args.datasets:
        if args.clean_data:
            import shutil
            shutil.rmtree(args.data_dir, ignore_errors=True)
            print(f"\nremoved synthetic data in {args.data_dir}")
        else:
            print(f"\nsynthetic data kept in {args.data_dir} so re-runs skip "
                  f"regeneration; pass --clean-data to remove it")

    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
