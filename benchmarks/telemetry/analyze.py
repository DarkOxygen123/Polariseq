"""Records in, tables out — with a gate in front that rejects bad measurements.

The gate is the point of this module. It is easy to write an analysis that
turns any pile of JSON into a confident table; the reason this one exists is
to refuse. It distinguishes two very different kinds of problem, because
conflating them makes the word "degraded" meaningless — measured: *every*
arm, ours and scanpy's alike, trips the sampler-blindness condition, since
numba and C extensions hold the GIL just as Rust does.

**Blocking** — the run does not support any claim:

- the machine swapped during it (wall time measured paging, not the library),
- the parent and child disagree about peak RSS by more than
  :data:`RSS_DISAGREEMENT_TOLERANCE` (two independent observers saw different
  things, so at least one missed the peak),
- or the arm failed, timed out, or was killed.

**Caveats** — the run is usable, but one *specific* series from it is not:

- the child sampler was blind for more than :data:`MAX_BLIND_FRACTION` of the
  run. This invalidates the in-process USS *curve* — it has holes exactly
  where the work happened — and nothing else. Peak memory and wall time come
  from the parent, which samples from outside the process where no GIL
  applies, so they stand. ``uss_series_usable`` says which figures may use
  the USS curve; the memory-over-time panel falls back to parent RSS.

Neither kind is ever dropped silently or quietly averaged in. A benchmark
that hides its own bad runs is how published numbers stop being reproducible.

Aggregation reports the whole distribution, not one number. Every row
carries a :class:`Stat` per metric — mean, sd, min, max, median, a 95% CI on
the mean, and `n` — because a single figure cannot say whether the machine
was quiet. Measured here: the same spill run gave 0.856 / 0.989 / 1.040 GB
across three trials of identical code, so a 2-trial summary can sit 20% off
a 5-trial one, and quoting it alone would be a confident-looking accident.

Published numbers therefore take the form `mean ± half-range (n=N)`, and any
`n` below :data:`MIN_TRIALS_FOR_STATS` is marked with `*` as under-powered.
The median is kept alongside the mean for the metrics where one bad run
should not move the headline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from .sampler import MAX_BLIND_FRACTION

#: Parent RSS and child RSS measure the same quantity from two sides. More
#: than this much disagreement means one of the samplers missed the peak.
RSS_DISAGREEMENT_TOLERANCE = 0.15

#: Steps that are harness scaffolding rather than pipeline work, and so are
#: excluded from `pipeline_s`. They remain in `wall_s` and in the step table,
#: because hiding them entirely would be its own kind of dishonesty.
_NON_PIPELINE_STEPS = frozenset({"warmup"})

_GB = 1e9


#: Trials below this make a mean meaningless and a spread misleading — with
#: two points the "range" is just the gap between two draws. Measured on
#: this machine: the heart_50k spill peak came out 0.856 / 0.989 / 1.040 GB
#: on three runs of identical code, so a 2-trial median can sit 20% off the
#: 5-trial one. Reports mark anything below this as under-powered rather
#: than quietly printing a confident-looking number.
MIN_TRIALS_FOR_STATS = 5


@dataclass
class Stat:
    """A measured quantity summarised across trials.

    Carries the whole shape, not one number: `mean` for the headline, `sd`
    for how noisy the machine was, `lo`/`hi` for the honest range a reader
    should expect to reproduce, and `n` so an under-powered result cannot be
    mistaken for a precise one.
    """

    n: int
    mean: float
    sd: float
    lo: float
    hi: float
    median: float

    @property
    def ci95(self) -> tuple[float, float]:
        """Normal-approximation 95% CI on the *mean* (not the spread).

        This narrows with more trials, which is what it should do: it
        answers "where is the true mean", not "what will my next run give".
        For the latter, quote `lo`/`hi`. With n < 3 it degenerates to the
        observed range rather than pretending to an interval.
        """
        if self.n < 3 or self.sd == 0.0:
            return (self.lo, self.hi)
        half = 1.96 * self.sd / math.sqrt(self.n)
        return (self.mean - half, self.mean + half)

    @property
    def underpowered(self) -> bool:
        return self.n < MIN_TRIALS_FOR_STATS

    @property
    def rel_spread(self) -> float:
        """(hi - lo) / mean — how reproducible this number is at all."""
        return (self.hi - self.lo) / self.mean if self.mean else float("nan")

    def format(self, unit: str = "", places: int = 2) -> str:
        """`mean ± half-range (n=N)`, the form every published number takes."""
        half = (self.hi - self.lo) / 2.0
        mark = "*" if self.underpowered else ""
        return (f"{self.mean:.{places}f} ± {half:.{places}f}{unit} "
                f"(n={self.n}{mark})")

    def as_dict(self) -> dict[str, Any]:
        lo, hi = self.ci95
        return {
            "n": self.n, "mean": self.mean, "sd": self.sd,
            "min": self.lo, "max": self.hi, "median": self.median,
            "ci95_lo": lo, "ci95_hi": hi,
            "underpowered": self.underpowered,
        }


def stat(values: Sequence[float]) -> Stat | None:
    """Summarise a list of trial results. ``None`` if there is nothing to say."""
    vals = [float(v) for v in values if v is not None and v == v]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return Stat(n=n, mean=mean, sd=sd, lo=min(vals), hi=max(vals),
                median=median(vals))


def quality(record: dict[str, Any]) -> dict[str, Any]:
    """Is this record fit to be quoted, and which of its series are usable?"""
    blocking: list[str] = []
    caveats: list[str] = []

    if record.get("status") != "ok":
        blocking.append(
            f"run status {record.get('status')!r}: {record.get('note', '')}".strip()
        )

    swapped = record.get("parent", {}).get("swap_out_delta_bytes", 0) or 0
    if swapped > 50_000_000:
        blocking.append(
            f"machine swapped {swapped / 1e6:.0f} MB during the run — wall "
            f"time includes paging and is not a measure of the library"
        )

    crss = record.get("peaks", {}).get("child_rss", 0)
    prss = record.get("peaks", {}).get("parent_rss", 0)
    if crss and prss:
        disagreement = abs(crss - prss) / max(crss, prss)
        if disagreement > RSS_DISAGREEMENT_TOLERANCE:
            blocking.append(
                f"parent and child disagree on peak RSS by "
                f"{100 * disagreement:.0f}% ({prss / _GB:.2f} vs "
                f"{crss / _GB:.2f} GB) — one sampler missed the peak"
            )

    gaps = record.get("child", {}).get("gaps", {}) or {}
    blind = gaps.get("blind_fraction")
    uss_usable = blind is not None and blind <= MAX_BLIND_FRACTION
    if not uss_usable:
        # Do not attribute this to the GIL. Measured 2026-09-08 on brain50k:
        # blind fraction was 23.9% at --threads 8, 1.6-2.5% at 4-6 threads,
        # and 1.7% at the library default, on identical code. Per-step
        # attribution put PCA — the longest step — at only 9.6% blind even in
        # the worst run, so `py.detach()` is releasing. The sampler is a
        # Python thread competing for a core with the kernel's worker threads,
        # so it starves when threads >= cores. Naming the GIL here sent an
        # earlier investigation down the wrong path.
        caveats.append(
            f"in-process sampler blind for {100 * (blind or 1.0):.1f}% of the "
            f"run (limit {100 * MAX_BLIND_FRACTION:.0f}%) — the USS *curve* "
            f"has holes where the sampler thread was not scheduled (usual "
            f"cause: threads >= cores, or a loaded machine); peaks and wall "
            f"time come from the parent sampler and are unaffected"
        )
    if not gaps.get("uss_available", True):
        caveats.append("USS unavailable (AccessDenied); only RSS was recorded")

    return {
        "ok": not blocking,
        "status": record.get("status"),
        "blocking": blocking,
        "caveats": caveats,
        "uss_series_usable": uss_usable,
        # Kept for callers that only want "is anything wrong here at all".
        "degraded": bool(blocking),
        "reasons": blocking + caveats,
    }


def _steps_of(record: dict[str, Any]) -> dict[str, float]:
    """`{step name: wall_s}` for one record, skipping unfinished steps."""
    return {
        st["name"]: st["wall_s"]
        for st in record.get("steps", [])
        if st.get("wall_s") is not None and st["wall_s"] == st["wall_s"]
    }


def _s(values: Sequence[float]) -> dict[str, Any] | None:
    """`stat()` as a plain dict, for embedding in a summary row."""
    st = stat(values)
    return st.as_dict() if st else None


def _group(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, bool], list[dict[str, Any]]]:
    """Group by (dataset, arm, with_umap).

    The UMAP flag is part of the key because a directory can legitimately hold
    both variants for one dataset — the ladder runs `--no-umap`, UMAP is
    measured separately at the small sizes — and averaging a with-UMAP wall
    time into a without-UMAP median is a silent lie, in whichever direction.
    """
    out: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for r in records:
        umap = bool(r.get("params", {}).get("with_umap", False))
        out.setdefault((r["dataset"]["label"], r["arm"], umap), []).append(r)
    return out


def _med(xs: list[float]) -> float:
    return median(xs) if xs else float("nan")


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, arm, UMAP variant): medians across repeats."""
    rows: list[dict[str, Any]] = []
    for (dataset, arm, umap), group in sorted(_group(records).items()):
        quals = [quality(r) for r in group]
        ok = [r for r, q in zip(group, quals, strict=True) if q["ok"]]
        walls = [r["wall_s"] for r in ok]
        blocking = sorted({x for q in quals for x in q["blocking"]})
        caveats = sorted({x for q in quals for x in q["caveats"]})
        reasons = blocking + caveats
        uss_usable = any(q["uss_series_usable"] for q in quals)
        steps: dict[str, float] = {}
        for name in {s["name"] for r in ok for s in r.get("steps", [])}:
            vals = [
                s["wall_s"]
                for r in ok
                for s in r.get("steps", [])
                if s["name"] == name and s["wall_s"] == s["wall_s"]
            ]
            if vals:
                steps[name] = round(_med(vals), 3)
        # `warmup` is the harness warming numba's JIT on scanpy's behalf. It
        # is our scaffolding, not scanpy's work, and charging it to scanpy
        # would invert the result: measured on heart_10k, including it made
        # scanpy-tuned (36.3 s) look *slower* than scanpy-default (33.4 s)
        # when its pipeline is in fact 5 s faster.
        pipeline_s = round(
            sum(v for k, v in steps.items() if k not in _NON_PIPELINE_STEPS), 3
        ) if steps else None

        rows.append(
            {
                "dataset": dataset,
                # What the tables print: the UMAP variant is visible in the
                # row itself, not inferred from which steps happen to exist.
                "dataset_display": f"{dataset} (+UMAP)" if umap else dataset,
                "umap": umap,
                "arm": arm,
                "pipeline_s": pipeline_s,
                # Carried onto the row so the scaling panel can plot against
                # dataset size without reopening the records.
                # Per-trial distributions, not just the median. A single
                # number hides whether the machine was quiet: the same
                # spill run measured 0.856-1.040 GB across three identical
                # trials, so the spread is part of the result, not noise to
                # be averaged away silently.
                "stats": {
                    "pipeline_s": _s(
                        [sum(v for k, v in _steps_of(r).items()
                             if k not in _NON_PIPELINE_STEPS) for r in ok]
                    ),
                    "wall_s": _s(walls),
                    "peak_rss_gb": _s([r["peaks"]["parent_rss"] / _GB for r in ok]),
                    "peak_uss_gb": _s([r["peaks"]["child_uss"] / _GB for r in ok]),
                    "steps": {
                        name: _s([_steps_of(r)[name] for r in ok if name in _steps_of(r)])
                        for name in {n for r in ok for n in _steps_of(r)}
                    },
                },
                "n_cells": group[0].get("dataset", {}).get("n_cells", 0),
                "n_genes": group[0].get("dataset", {}).get("n_genes", 0),
                "nnz": group[0].get("dataset", {}).get("nnz", 0),
                "arm_label": group[0].get("arm_label", arm),
                "n_reps": len(group),
                "n_ok": len(ok),
                "status": group[0].get("status") if not ok else "ok",
                "wall_s": round(_med(walls), 3) if walls else None,
                "wall_spread_s": round(max(walls) - min(walls), 3) if len(walls) > 1 else 0.0,
                "peak_rss_gb": round(_med([r["peaks"]["parent_rss"] for r in ok]) / _GB, 3)
                if ok else None,
                "peak_uss_gb": round(_med([r["peaks"]["child_uss"] for r in ok]) / _GB, 3)
                if ok else None,
                "baseline_uss_gb": round(
                    _med([r["child"].get("baseline", {}).get("uss", 0) for r in ok]) / _GB, 3
                ) if ok else None,
                "spill_peak_gb": round(
                    _med([r.get("disk", {}).get("spill_peak_bytes", 0) for r in ok]) / _GB, 3
                ) if ok else None,
                "steps": steps,
                "degraded": bool(blocking),
                "blocking": blocking,
                "caveats": caveats,
                "reasons": reasons,
                "uss_series_usable": uss_usable,
                "n_clusters": group[0].get("outputs", {}).get("n_clusters"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], step_order: tuple[str, ...] = ()) -> str:
    """The headline table: wall and memory per arm, with a degradation column.

    RSS and USS are shown side by side and always both, because each is
    misleading alone: RSS counts memory-mapped page cache the process does
    not really own (it overstates an out-of-core implementation), and USS
    ignores page cache the workload's speed nonetheless depends on (it
    understates one). A reader who only trusts one of them can still read
    this table; a writer who quotes only one cannot.
    """
    head = (
        "| dataset | arm | pipeline (s) | wall (s) | peak RSS (GB) | "
        "peak USS (GB) | baseline USS (GB) | spill (GB) | clusters | quality |"
    )
    sep = "|" + "---|" * 10
    lines = [head, sep]
    for r in rows:
        if r["degraded"]:
            q = "BLOCKED"
        elif r.get("caveats"):
            q = "ok*"
        else:
            q = "ok"
        def f(x: Any, nd: int = 2) -> str:
            return "—" if x is None else f"{x:.{nd}f}"
        lines.append(
            f"| {r['dataset_display']} | {r['arm_label']} | {f(r.get('pipeline_s'), 1)} | "
            f"{f(r['wall_s'], 1)} | "
            f"{f(r['peak_rss_gb'])} | {f(r['peak_uss_gb'])} | "
            f"{f(r['baseline_uss_gb'])} | {f(r['spill_peak_gb'])} | "
            f"{r['n_clusters'] if r['n_clusters'] is not None else '—'} | {q} |"
        )
    blocked = [r for r in rows if r.get("blocking")]
    if blocked:
        lines.append("")
        lines.append("**Runs that support no claim:**")
        for r in blocked:
            for reason in r["blocking"]:
                lines.append(f"- `{r['dataset_display']}` / `{r['arm']}`: {reason}")
    caveated = [r for r in rows if r.get("caveats")]
    if caveated:
        lines.append("")
        lines.append("**Caveats (the run stands; one series from it does not):**")
        for r in caveated:
            for reason in r["caveats"]:
                lines.append(f"- `{r['dataset_display']}` / `{r['arm']}`: {reason}")
    if step_order:
        lines.append("")
        lines.append(step_table(rows, step_order))
    return "\n".join(lines)


def step_table(rows: list[dict[str, Any]], step_order: tuple[str, ...]) -> str:
    """Per-step wall time, median across repeats. Missing steps show as `—`."""
    present = [s for s in step_order if any(s in r["steps"] for r in rows)]
    lines = [
        "| dataset | arm | " + " | ".join(present) + " |",
        "|" + "---|" * (len(present) + 2),
    ]
    for r in rows:
        cells = [
            f"{r['steps'][s]:.2f}" if s in r["steps"] else "—" for s in present
        ]
        lines.append(f"| {r['dataset_display']} | {r['arm_label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def speedup_table(rows: list[dict[str, Any]], ours: str = "polariseq-ram") -> str:
    """Our wall time against each scanpy configuration.

    Both baselines are shown. Quoting the ratio against `scanpy-default`
    alone is the single easiest way to overstate a result here, since most of
    that gap is scanpy's `n_jobs=1` default and its `leidenalg` flavour
    rather than anything about the kernels.
    """
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["dataset_display"], {})[r["arm"]] = r
    lines = [
        "| dataset | ours, pipeline (s) | vs scanpy-default | vs scanpy-tuned |",
        "|---|---|---|---|",
    ]
    for dataset, arms in sorted(by.items()):
        mine = arms.get(ours, {}).get("pipeline_s")
        if not mine:
            continue
        def ratio(name: str, arms: dict = arms, mine: float = mine) -> str:
            other = arms.get(name, {}).get("pipeline_s")
            return f"{other / mine:.2f}x" if other and mine else "—"
        lines.append(
            f"| {dataset} | {mine:.1f} | {ratio('scanpy-default')} | {ratio('scanpy-tuned')} |"
        )
    return "\n".join(lines)
