#!/usr/bin/env python3
"""Run the telemetry comparison and render its tables and figures.

Examples
--------
Quick check that everything is wired up (seconds, on the tiny fixture)::

    python benchmarks/run_telemetry.py --datasets fixture --quick

The real comparison on one dataset, three arms, two repeats::

    python benchmarks/run_telemetry.py --datasets heart_10k --reps 2

The full ladder, including the out-of-core arm, then figures::

    python benchmarks/run_telemetry.py --datasets heart_10k,heart_50k,heart_100k \\
        --arms polariseq-ram,polariseq-spill,scanpy-default,scanpy-tuned --reps 2
    python benchmarks/run_telemetry.py --report-only

Results are JSON under ``benchmarks/results/telemetry/``; each run is a
separate file, so an interrupted campaign resumes by simply running the rest.
Nothing is ever overwritten and nothing is deleted, because a result you
cannot go back to is a result you have to re-earn.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from telemetry.analyze import markdown_table, speedup_table, summarize  # noqa: E402
from telemetry.arms import ARMS, CANONICAL_STEPS, DEFAULT_ARMS, dataset_facts  # noqa: E402
from telemetry.equivalence import agreement_floors, compare_all  # noqa: E402
from telemetry.record import Params, load_records  # noqa: E402
from telemetry.runner import run_one  # noqa: E402

REPO = HERE.parent
RESULTS = HERE / "results" / "telemetry"

#: Known datasets. Paths are resolved against the repo root; anything missing
#: is reported and skipped rather than failing the campaign.
#: Cache for the public ladder and everything derived from it. `validate.py`
#: downloads here, and `make_subsets.py` writes the brain* subsets here, so one
#: directory holds every input a campaign reads.
_VD = "benchmarks/results/validation_data"

DATASETS = {
    "fixture": "tests/data/medium_10kx2k.h5ad",
    # Tier 1 — public 10x, downloaded by validate.py.
    "pbmc3k": f"{_VD}/pbmc3k.h5ad",
    "pbmc10k": f"{_VD}/pbmc10k.h5ad",
    "neuron10k": f"{_VD}/neuron10k.h5ad",
    "pbmc20k": f"{_VD}/pbmc20k.h5ad",
    # Tiers 2-3 — subsets of the 10x 1.3M mouse brain, built by
    # make_subsets.py with a recorded seed. Provenance travels in each
    # file's `uns`.
    "brain5k": f"{_VD}/brain5k.h5ad",
    "brain50k": f"{_VD}/brain50k.h5ad",
    "brain100k": f"{_VD}/brain100k.h5ad",
    "brain200k": f"{_VD}/brain200k.h5ad",
    "brain400k": f"{_VD}/brain400k.h5ad",
    "brain800k": f"{_VD}/brain800k.h5ad",
    "brain300k": f"{_VD}/brain300k.h5ad",
    "brain600k": f"{_VD}/brain600k.h5ad",
    "brain1000k": f"{_VD}/brain1000k.h5ad",
    "brain1306k": f"{_VD}/brain1306k.h5ad",
    # Tier 4 — the human fetal atlas (Cao et al. 2020, GEO GSE156793), 4.06M
    # cells, converted and subset (nested, seed 0) by fetal_prepare.py.
    # gzip-compressed h5ad, read as such by every arm.
    "fetal1000k": "benchmarks/results/fetal_data/fetal1000k.h5ad",
    "fetal2000k": "benchmarks/results/fetal_data/fetal2000k.h5ad",
    "fetal4062k": "benchmarks/results/fetal_data/fetal4062k.h5ad",
    # Tier 3 — the full 1.3M file, read natively as 10x HDF5.
    "brain1300k": f"{_VD}/1M_neurons_filtered_gene_bc_matrices_h5.h5",
    # Legacy internal ladder, absent from this machine; kept so old records
    # still resolve by name.
    "heart_10k": "data/bench/heart_10k.h5ad",
    "heart_50k": "data/bench/heart_50k.h5ad",
    "heart_100k": "data/bench/heart_100k.h5ad",
    "combat_10k": "data/bench/combat_10k.h5ad",
    "combat_50k": "data/bench/combat_50k.h5ad",
    "combat_100k": "data/bench/combat_100k.h5ad",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="heart_10k",
                   help="comma-separated names from DATASETS, or explicit .h5ad paths")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help=f"comma-separated; known: {','.join(sorted(ARMS))}")
    p.add_argument("--reps", type=int, default=2,
                   help="repeats per (dataset, arm); medians are reported")
    p.add_argument("--threads", type=int, default=0,
                   help="threads for every arm (0 = detect and record)")
    p.add_argument("--ram-gb", type=float, default=6.0,
                   help="RAM ceiling pinned into every arm's budget (0 = detect "
                        "per process; pinned makes spill block sizing reproducible)")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--no-umap", action="store_true",
                   help="skip UMAP (it dominates wall time on large inputs)")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    p.add_argument("--memory-limit-gb", type=float, default=None,
                   help="stop a run whose process footprint passes this (macOS) and "
                        "record it as exceeds_memory")
    p.add_argument("--interval", type=float, default=0.02, help="child sampling period (s)")
    p.add_argument("--cache", choices=("warm", "as-is"), default="warm",
                   help="page-cache treatment before each run; the warm read "
                        "is timed and recorded so residency is checkable")
    p.add_argument("--order", choices=("rotate", "fixed"), default="rotate",
                   help="rotate the arm order per repeat so no arm always "
                        "runs first (default), or keep the given order")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--quick", action="store_true",
                   help="tiny parameters for a wiring check, not for any claim")
    p.add_argument("--report-only", action="store_true",
                   help="skip running; re-render tables and figures from existing records")
    p.add_argument("--figures", action="store_true", help="render figure panels")
    p.add_argument("--warmup", action="store_true",
                   help="run each (dataset, arm) once and DISCARD it before "
                        "the timed reps. Warms the Rust extension's dlopen, "
                        "the Python import tree, numba's disk cache and the "
                        "data page cache equally for every arm. Measured "
                        "2026-09-09: without it, polariseq-ram rep1 on pbmc3k "
                        "was 68% slower than later reps (cold dlopen of the "
                        "6.4 MB extension); scanpy showed no such effect "
                        "because its tuned arm already warms its own JIT.")
    p.add_argument("--gate", choices=("off", "calibrate", "default"), default="off",
                   help="wait for a quiet machine before each run. 'calibrate' "
                        "derives the band from this machine's idle state "
                        "(recommended); 'default' uses fixed limits. Measured: "
                        "machine load alone moves wall time 2x.")
    p.add_argument("--gate-timeout", type=float, default=1800.0,
                   help="seconds to wait for the machine to settle before "
                        "recording the run anyway (default 30 min)")
    p.add_argument("--gate-pressured-fraction", type=float, default=1.0,
                   help="for runs that need more memory than the machine has free at rest, "
                        "start once available memory reaches this fraction of the at-rest "
                        "level (default 1.0: the at-rest level itself)")
    p.add_argument("--reference", default="scanpy-tuned",
                   help="arm the equivalence check compares against")
    p.add_argument("--resume", action="store_true",
                   help="skip every (dataset, arm, repeat) whose outcome is already "
                        "in --results at these parameters, and the remaining repeats "
                        "of any (dataset, arm) that failed. Runs recorded with "
                        "GATE NOT MET are run again.")
    return p


def _variant(params: dict) -> tuple:
    """The parameters that make two records the same measurement."""
    return tuple(params.get(k) for k in (
        "n_hvg", "n_pcs", "n_neighbors", "min_genes", "with_umap", "threads", "ram_gb"))


def on_record(results_dir: Path, params: Params) -> tuple[set, set]:
    """What `--resume` may skip: (done, failed).

    `done` holds (label, arm, rep) for every run at these parameters that
    either completed with its gate met or failed outright. A completion
    measured through interference (gate_ok False) is not done: it cannot
    enter a median, so resuming runs it again. `failed` holds (label, arm)
    for anything that did not complete — failures on this ladder are memory
    limits, so the remaining repeats would fail the same way, at up to the
    full timeout each.
    """
    want = _variant(params.__dict__)
    done: set = set()
    failed: set = set()
    for r in load_records(results_dir):
        if _variant(r.get("params", {})) != want:
            continue
        key = (str(r["dataset"]["label"]), r["arm"])
        if r.get("status") != "ok":
            failed.add(key)
            done.add((*key, int(r.get("rep", 0))))
        elif (r.get("disk") or {}).get("gate_ok") is not False:
            done.add((*key, int(r.get("rep", 0))))
    return done, failed


def resolve(name: str) -> tuple[str, str] | None:
    if name in DATASETS:
        path = REPO / DATASETS[name]
        return (name, str(path)) if path.exists() else None
    p = Path(name)
    if p.exists():
        return (p.stem, str(p))
    return None


def collect_predictions(records: list[dict[str, object]], *,
                        with_umap: bool) -> dict[str, list[tuple[int, float]]]:
    """``ps.plan()``'s predicted peak memory per Polariseq arm, for panel D.

    A budget predictor earns its place by *bracketing* measurement across the
    whole ladder, so the prediction line belongs on the scaling panel next to
    the measured points. Only arms that actually ran are predicted, and the
    plan matches the figure's UMAP variant — UMAP changes the footprint.
    The prediction is expected to err high (a budget should); chapter 09
    reports measured-vs-predicted so the direction of the error is visible.
    """
    try:
        from polariseq import scan_h5ad_meta
        from polariseq.planning import estimate as _estimate
        from polariseq.resources import Budget
    except ImportError as exc:  # pragma: no cover - report-only without the wheel
        print(f"[predict] polariseq not importable, skipping prediction line: {exc}",
              file=sys.stderr)
        return {}
    storages = {"polariseq-ram": "ram", "polariseq-spill": "spill"}
    arms = {a for a in storages if any(r["arm"] == a for r in records)}
    if not arms:
        return {}
    # label -> the inputs a plan needs, *including the budget the campaign
    # actually pinned*. `ps.plan()` reads whatever `ps.budget()` is at the
    # moment this function runs, which is the ambient auto-detected budget on
    # this machine right now -- almost never what the campaign used, and
    # exactly the "floating budget" failure this harness exists to catch
    # elsewhere (chapter 10, rule 4). Measured: a campaign pinned to 4 GB
    # rendered a prediction against an ambient 2 GB budget when the report
    # was regenerated later, and the resulting pred/meas ratios inverted
    # between 10k (3.1x, over) and 100k (0.4x, *under* -- which is the
    # dangerous direction for a memory budget) for no reason but that.
    #
    # So the prediction is built from `estimate()` directly, against a
    # `Budget` reconstructed from each record's own `params.ram_gb` /
    # `params.threads`, never from the ambient one.
    seen: dict[str, tuple[str, int, int, float, int]] = {}
    for r in records:
        d = r["dataset"]
        params = r.get("params", {})
        seen[str(d["label"])] = (
            str(d.get("path", "")),
            int(params.get("n_hvg", 2000)),
            int(params.get("n_pcs", 50)),
            float(params.get("ram_gb", 0.0)),
            int(params.get("threads", 0)) or 1,
        )
    preds: dict[str, list[tuple[int, float]]] = {a: [] for a in arms}
    for label, (path, n_hvg, n_pcs, ram_gb, threads) in sorted(seen.items()):
        if not path or not Path(path).exists():
            print(f"[predict] no resolvable path for {label}, skipping", file=sys.stderr)
            continue
        if ram_gb <= 0:
            print(f"[predict] {label}: no ram_gb on record (unpinned campaign?), "
                  f"skipping prediction", file=sys.stderr)
            continue
        try:
            meta = scan_h5ad_meta(path)
        except Exception as exc:  # noqa: BLE001 - a failed scan must not sink the figure
            print(f"[predict] {label}: {exc}", file=sys.stderr)
            continue
        b = Budget(ram_bytes=int(ram_gb * 1e9), threads=threads,
                   spill_dir="/tmp", spill_bytes=int(500e9))
        for arm in sorted(arms):
            try:
                p = _estimate(int(meta["n_cells"]), int(meta["n_genes"]), int(meta["nnz"]), b,
                              n_hvg=n_hvg, n_pcs=n_pcs, storage=storages[arm],
                              with_umap=with_umap)
            except Exception as exc:  # noqa: BLE001 - a failed plan must not sink the figure
                print(f"[predict] {arm} on {label}: {exc}", file=sys.stderr)
                continue
            preds[arm].append((int(meta["n_cells"]), float(p.peak_bytes) / 1e9))
    return {a: pts for a, pts in preds.items() if pts}



def _peak_requirements(results_dir, cap_fraction: float = 0.80) -> dict:
    """Peak memory each (dataset, arm) has been observed to need.

    The load gate's calibrated floor describes the machine at rest, not the
    run about to start, so on a busy machine it adapts downward and admits
    runs it cannot hold. Prior records are the only honest estimate of what a
    run needs, and every combination in a campaign has usually been measured
    at least once before reaching the sizes that matter.

    The requirement is the observed peak with no margin added, capped at
    `cap_fraction` of installed RAM. The cap matters on the 8 GiB machine,
    where an arm peaking at 6.8 GB cannot leave room for the OS as well: a
    floor above the cap would never be met, so the gate would spend its whole
    timeout waiting and then run anyway, converting a memory problem into a
    lost afternoon. Above the cap the run is inherently pressured on this
    hardware, which is a fact about the machine worth recording rather than
    waiting out.

    A combination with no history contributes nothing and is gated on the
    calibrated floor alone. That is the first run at a new size, which is
    exactly the run whose requirement nobody knows yet.
    """
    import glob
    import json
    from collections import defaultdict

    peaks: dict = defaultdict(int)
    for f in glob.glob(str(Path(results_dir) / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if d.get("status") != "ok":
            continue
        key = (d.get("dataset", {}).get("label"), d.get("arm"))
        # Demand, not RSS. RSS omits whatever the OS compressed, so gating on
        # it sets the bar below what the run actually needs: measured
        # 2026-09-10, polariseq-ram at 600K cleared a gate built from its
        # 4.18 GB RSS while demanding 9.35 GB, and its wall time then varied
        # 64-142 s with machine state. Falls back to RSS for records taken
        # before compression accounting existed.
        pa = d.get("parent") or {}
        need = pa.get("peak_demand_bytes") or (d.get("peaks") or {}).get("parent_rss", 0)
        if key[0] and key[1] and need:
            peaks[key] = max(peaks[key], int(need))

    import psutil

    cap = int(psutil.virtual_memory().total * cap_fraction)
    return {k: min(v, cap) for k, v in peaks.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = Path(args.results)

    params = Params(
        n_hvg=100 if args.quick else args.n_hvg,
        n_pcs=10 if args.quick else args.n_pcs,
        n_neighbors=args.n_neighbors,
        min_genes=args.min_genes,
        with_umap=not (args.no_umap or args.quick),
        threads=args.threads,
        ram_gb=args.ram_gb,
    )

    if not args.report_only:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            print(f"unknown arms: {unknown}; known: {sorted(ARMS)}", file=sys.stderr)
            return 2

        warmed: set[tuple[str, str]] = set()
        band = None
        if args.gate != "off":
            from telemetry.loadgate import Band, calibrate
            band = calibrate() if args.gate == "calibrate" else Band()

        needs = _peak_requirements(results)
        done, failed = on_record(results, params) if args.resume else (set(), set())

        targets = []
        for name in (d.strip() for d in args.datasets.split(",") if d.strip()):
            got = resolve(name)
            if got is None:
                print(f"[skip] dataset not found: {name}", file=sys.stderr)
                continue
            targets.append(got)
        if not targets:
            print("no datasets to run", file=sys.stderr)
            return 2

        total = len(targets) * len(arms) * args.reps
        n_run = 0
        for label, path in targets:
            facts = dataset_facts(path)
            print(f"\n=== {label}: {facts['n_cells']:,} x {facts['n_genes']:,}, "
                  f"{facts['nnz']:,} nnz, {facts['file_bytes'] / 1e9:.2f} GB file ===")
            for rep in range(1, args.reps + 1):
                # Rotate the arm order every repeat. Whichever arm runs first
                # after the file is warmed sees the freshest cache and the
                # quietest machine; running them in a fixed order gives that
                # advantage to the same arm every time, which is a systematic
                # bias rather than noise the repeats average out. Rotation
                # makes each arm take each position in turn.
                ordered = (
                    arms[(rep - 1) % len(arms):] + arms[: (rep - 1) % len(arms)]
                    if args.order == "rotate"
                    else arms
                )
                for arm in ordered:
                    n_run += 1
                    if (label, arm, rep) in done:
                        print(f"[{n_run}/{total}] {label} / {arm} rep{rep} "
                              f"— on record, skipped")
                        continue
                    if (label, arm) in failed:
                        print(f"[{n_run}/{total}] {label} / {arm} rep{rep} "
                              f"— failed earlier at this size, skipped")
                        continue
                    print(f"[{n_run}/{total}] {label} / {arm} rep{rep} ...",
                          end="", flush=True)
                    if args.warmup and (label, arm) not in warmed:
                        # Discarded: results go to a temp dir that is thrown
                        # away, so nothing from a cold start reaches the
                        # archive. See --warmup for why this exists.
                        print(f"  warming {label} / {arm} ...",
                              end="", flush=True)
                        with tempfile.TemporaryDirectory() as tmp:
                            run_one(
                                arm=arm, path=path, label=label, params=params,
                                results_dir=Path(tmp), rep=0,
                                timeout_s=args.timeout,
                                interval_s=args.interval, cache=args.cache,
                                facts=facts, gate=None, keep_artifacts=False,
                            )
                        warmed.add((label, arm))
                        print(" done")

                    rec = run_one(
                        arm=arm, path=path, label=label, params=params,
                        results_dir=results, rep=rep, timeout_s=args.timeout,
                        memory_limit_bytes=(int(args.memory_limit_gb * 1e9)
                                            if args.memory_limit_gb else None),
                        interval_s=args.interval, cache=args.cache, facts=facts,
                        gate=band, gate_timeout_s=args.gate_timeout,
                        needs_available_bytes=needs.get((label, arm), 0),
                        pressured_fraction=args.gate_pressured_fraction,
                        # Artifacts are the PCA embedding, labels, HVG list and
                        # cell names, and equivalence needs one set per
                        # (dataset, arm) rather than one per repeat. Writing
                        # them every repeat cost 213 MB per brain200k run and
                        # filled the disk mid-campaign on 2026-09-09, which
                        # then surfaced as failed runs that looked like
                        # capacity limits.
                        keep_artifacts=(rep == 1),
                    )
                    print(
                        f" {rec.status} wall={rec.wall_s:.1f}s "
                        f"rss={rec.peaks['parent_rss'] / 1e9:.2f}GB "
                        f"uss={rec.peaks['child_uss'] / 1e9:.2f}GB"
                        + (f" — {rec.note[:120]}" if rec.note else "")
                    )
                    if args.resume and rec.status != "ok":
                        failed.add((label, arm))

    records = load_records(results)
    if not records:
        print("no records to report", file=sys.stderr)
        return 1
    rows = summarize(records)

    print("\n## Wall time and memory\n")
    print(markdown_table(rows, CANONICAL_STEPS))
    print("\n## Speedup\n")
    print(speedup_table(rows))

    print("\n## Do the arms agree on the answer?\n")
    comparisons = compare_all(records, reference=args.reference,
                              n_neighbors=args.n_neighbors)
    all_floors = agreement_floors(records, reference=args.reference,
                                  n_neighbors=args.n_neighbors)
    floors = {k: v["umap_knn_overlap"] for k, v in all_floors.items()
              if "umap_knn_overlap" in v}
    if all_floors:
        print("Seed-to-seed floors — the reference arm against itself at a "
              "second seed. A cross-implementation score at or above the "
              "floor is as much agreement as the metric admits on that "
              "dataset; these replace the fixed thresholds where present:")
        for label, f in sorted(all_floors.items()):
            parts = [f"{k} **{v:.3f}**" for k, v in sorted(f.items())]
            print(f"- {label}: " + ", ".join(parts))
        print()
    if not comparisons:
        print(f"(no `{args.reference}` reference run available to compare against)")
    for c in comparisons:
        checks = ", ".join(f"{k}={v}" for k, v in c["checks"].items())
        line = (f"- **{c['dataset']}** {c['arm']} vs {c['reference']}: "
                f"**{c['verdict']}** — {checks}")
        # Every overlap is printed next to its floor, so neither number can
        # be quoted alone and mistaken for the whole story.
        if "umap_knn_overlap" in c["checks"] and c["dataset"] in floors:
            line += f" (seed-to-seed floor: {floors[c['dataset']]:.3f})"
        print(line)
        for s in c["skipped"]:
            print(f"  - not checked: {s}")

    if args.figures:
        from telemetry.figures import make_panels

        # One figure per (dataset, UMAP variant): the ladder's --no-umap spine
        # and the separately-measured UMAP runs are different pipelines and
        # must not share panels or medians. Panel D and the prediction line
        # span the whole ladder (all datasets of that variant) in every
        # figure; panels A-C are per-dataset and filter internally.
        for umap in sorted({bool(r.get("params", {}).get("with_umap", False))
                            for r in records}):
            recs_v = [r for r in records
                      if bool(r.get("params", {}).get("with_umap", False)) == umap]
            rows_v = [r for r in rows if r["umap"] == umap]
            preds = collect_predictions(recs_v, with_umap=umap)
            for label in sorted({str(r["dataset"]["label"]) for r in recs_v}):
                display = f"{label} (+UMAP)" if umap else label
                suffix = "_umap" if umap else ""
                out = make_panels(
                    recs_v, rows_v, label,
                    results.parent / "figures" / f"telemetry_{label}{suffix}.png",
                    CANONICAL_STEPS, predictions=preds, display=display,
                )
                print(f"\nwrote {out}"
                      + (f" (prediction line: {', '.join(sorted(preds))})"
                         if preds else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
