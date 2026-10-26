"""Benchmark panels, drawn only from what the records actually support.

Four panels, each answering one question:

- **A — memory over time.** The panel the whole harness exists for. It shows
  that Polariseq's footprint *moves* — rising for a block, falling when the
  block is released — while an in-memory implementation holds a plateau for
  the length of the run. Drawn from the **parent RSS** series by default,
  because that one is GIL-immune; the in-process USS series is overlaid only
  for runs whose ``uss_series_usable`` flag is set, and the caption says
  which is which.

- **B — where the time goes.** Per-step wall time. A single end-to-end number
  hides that most of a scanpy run is two steps, and hides which of our steps
  is now the bottleneck.

- **C — peak cost.** RSS and USS side by side for every arm, plus the spill
  file for out-of-core arms, so the RAM-for-SSD trade is visible rather than
  implied.

- **D — scaling.** Peak memory against cell count across the dataset ladder,
  with ``ps.plan()``'s prediction drawn as a line. A predictor that tracks
  measurement across a decade of dataset sizes is a much stronger claim than
  any single benchmark row.

Style rules that are not negotiable for a figure meant to be believed: every
axis labelled with units, every arm the same colour in every panel, the
machine and commit stamped on the figure itself, and no truncated y-axis on
a bar chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: One colour per arm, everywhere. Chosen to stay distinguishable in
#: greyscale and for the common forms of colour blindness (Okabe-Ito).
ARM_COLORS = {
    "polariseq-ram": "#0072B2",
    "polariseq-spill": "#56B4E9",
    "scanpy-default": "#D55E00",
    "scanpy-tuned": "#E69F00",
}
ARM_ORDER = ("polariseq-ram", "polariseq-spill", "scanpy-tuned", "scanpy-default")

_GB = 1e9


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
        }
    )
    return plt


def _stamp(fig: Any, records: list[dict[str, Any]]) -> None:
    """Put the provenance on the figure, not just in the caption."""
    if not records:
        return
    env = records[0].get("env", {})
    fig.text(
        0.005, 0.005,
        f"{env.get('cpu', '?')} · {env.get('total_ram_bytes', 0) / _GB:.1f} GB RAM · "
        f"polariseq {env.get('git_sha', '?')} · scanpy "
        f"{env.get('versions', {}).get('scanpy', '?')}",
        fontsize=6, color="#666666", ha="left", va="bottom",
    )


def _series(rec: dict[str, Any], prefer: str = "parent_rss"
            ) -> tuple[list[float], list[float], str]:
    """The memory time-series to plot, and an honest label for it."""
    from .analyze import quality

    if prefer == "child_uss" and quality(rec)["uss_series_usable"]:
        s = rec.get("child", {}).get("samples", {})
        return s.get("t", []), [x / _GB for x in s.get("uss", [])], "USS (in-process)"
    s = rec.get("parent", {}).get("samples", {})
    return s.get("t", []), [x / _GB for x in s.get("rss", [])], "RSS (external sampler)"


def panel_memory_over_time(records: list[dict[str, Any]], ax: Any, *,
                           dataset: str, prefer: str = "parent_rss",
                           display: str | None = None) -> None:
    """Panel A. One line per arm, with step boundaries marked.

    Parent RSS is the spine (GIL-immune); the in-process USS series is
    overlaid dashed for runs whose sampler was not blind through it — after
    the bindings released the GIL, ours are not, and the private-memory curve
    is the whole point of the library.
    """
    from .analyze import quality

    plotted = [r for r in records if r["dataset"]["label"] == dataset and r["status"] == "ok"]
    plotted.sort(key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)
    seen: set[str] = set()
    for rec in plotted:
        if rec["arm"] in seen:
            continue  # one representative run per arm; repeats live in the tables
        seen.add(rec["arm"])
        t, y, label = _series(rec, prefer)
        if not t:
            continue
        color = ARM_COLORS.get(rec["arm"], "#444444")
        ax.plot(t, y, lw=1.2, color=color,
                label=f"{rec.get('arm_label', rec['arm'])} — {label}")
        if quality(rec)["uss_series_usable"]:
            tu, yu, _ = _series(rec, "child_uss")
            if tu:
                ax.plot(tu, yu, lw=0.9, ls="--", color=color, alpha=0.8,
                        label=f"{rec.get('arm_label', rec['arm'])} — USS (in-process)")
        for step in rec.get("steps", []):
            if step.get("t_end") and step["wall_s"] > 0.05 * max(t):
                ax.axvline(step["t_begin"], color=color,
                           lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("memory (GB)")
    ax.set_title(f"A · Memory over the workflow — {display or dataset}", loc="left", fontsize=10)
    ax.legend(fontsize=7)


def panel_step_times(rows: list[dict[str, Any]], ax: Any, *, dataset: str,
                     step_order: tuple[str, ...], display: str | None = None) -> None:
    """Panel B. Grouped bars with min–max whiskers, log y.

    Step costs span three orders of magnitude, hence log y. The whiskers are
    the observed range across trials, not a standard error: a reader wants to
    know what their own next run will do, and the range answers that where a
    shrinking confidence interval does not.
    """
    import numpy as np

    sel = [r for r in rows if r["dataset"] == dataset]
    sel.sort(key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)
    steps = [s for s in step_order if any(s in r["steps"] for r in sel)]
    x = np.arange(len(steps))
    width = 0.8 / max(1, len(sel))
    for i, r in enumerate(sel):
        # A step this arm never reported gets no bar at all. Clamping an
        # absent step to the log floor draws a visible near-zero bar, which
        # reads as "measured, and it was instant" — the out-of-core arm
        # fuses six steps into one `preprocess`, and rendering it as six
        # ~1 ms bars would be a straightforward lie about what was measured.
        vals = [max(r["steps"][s], 1e-3) if s in r["steps"] else np.nan for s in steps]
        st = (r.get("stats") or {}).get("steps") or {}
        lo, hi = [], []
        for k, name in enumerate(steps):
            d = st.get(name) or {}
            base = vals[k]
            if base != base:  # NaN — nothing to whisker
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            lo.append(max(d.get("min", base), 1e-3))
            hi.append(max(d.get("max", base), 1e-3))
        err = [
            [0.0 if v != v else max(0.0, v - a) for v, a in zip(vals, lo, strict=True)],
            [0.0 if v != v else max(0.0, b - v) for v, b in zip(vals, hi, strict=True)],
        ]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               yerr=err, capsize=2, error_kw={"lw": 0.7, "ecolor": "#333333"},
               label=r["arm_label"], color=ARM_COLORS.get(r["arm"], "#444444"))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.set_ylabel("wall time (s, log)")
    ax.set_title(f"B · Time per step — {display or dataset}", loc="left", fontsize=10)
    ax.legend(fontsize=7)


def panel_peak_cost(rows: list[dict[str, Any]], ax: Any, *, dataset: str,
                    display: str | None = None) -> None:
    """Panel C. Peak RSS and USS together, plus spill, with nothing hidden."""
    import numpy as np

    sel = [r for r in rows if r["dataset"] == dataset]
    sel.sort(key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)
    x = np.arange(len(sel))
    rss = [r["peak_rss_gb"] or 0 for r in sel]
    uss = [r["peak_uss_gb"] or 0 for r in sel]
    spill = [r["spill_peak_gb"] or 0 for r in sel]
    def _err(metric: str, mid: list[float]) -> list[list[float]]:
        lo, hi = [], []
        for r, m in zip(sel, mid, strict=True):
            st = (r.get("stats") or {}).get(metric) or {}
            lo.append(max(0.0, m - st.get("min", m)))
            hi.append(max(0.0, st.get("max", m) - m))
        return [lo, hi]

    ekw: dict[str, Any] = {"capsize": 2, "error_kw": {"lw": 0.7, "ecolor": "#333333"}}
    ax.bar(x - 0.22, rss, 0.2, label="peak RSS", color="#888888",
           yerr=_err("peak_rss_gb", rss), **ekw)
    ax.bar(x, uss, 0.2, label="peak USS",
           color=[ARM_COLORS.get(r["arm"], "#444") for r in sel],
           yerr=_err("peak_uss_gb", uss), **ekw)
    if any(spill):
        ax.bar(x + 0.22, spill, 0.2, label="spill file on SSD",
               color="none", edgecolor="#000000", hatch="///", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["arm_label"] for r in sel], rotation=20, ha="right")
    ax.set_ylabel("peak (GB)")
    ax.set_ylim(bottom=0)  # never truncate a bar chart's baseline
    ax.set_title(f"C · Peak cost — {display or dataset}", loc="left", fontsize=10)
    ax.legend(fontsize=7)


def panel_scaling(rows: list[dict[str, Any]], ax: Any, *,
                  predictions: dict[str, list[tuple[int, float]]] | None = None) -> None:
    """Panel D. Peak memory vs cell count, with the planner's prediction."""
    by_arm: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        n = r.get("n_cells") or 0
        if n and r["peak_uss_gb"]:
            by_arm.setdefault(r["arm"], []).append((n, r["peak_uss_gb"]))
    for arm, pts in sorted(by_arm.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, lw=1.2,
                color=ARM_COLORS.get(arm, "#444444"), label=arm)
    for arm, pts in (predictions or {}).items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", lw=1.0,
                color=ARM_COLORS.get(arm, "#444444"), alpha=0.7,
                label=f"{arm} — ps.plan() prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel("peak USS (GB)")
    ax.set_title("D · Scaling, measured vs predicted", loc="left", fontsize=10)
    ax.legend(fontsize=7)


def make_panels(records: list[dict[str, Any]], rows: list[dict[str, Any]],
                dataset: str, out_path: str | Path,
                step_order: tuple[str, ...],
                predictions: dict[str, list[tuple[int, float]]] | None = None,
                display: str | None = None) -> Path:
    """Render the four-panel figure for one dataset.

    ``dataset`` is the raw label records are filtered on; ``display`` is what
    the titles show (e.g. ``heart_10k (+UMAP)`` when a UMAP-on variant is
    being rendered separately from the ladder's ``--no-umap`` spine).
    """
    plt = _style()
    shown = display or dataset
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panel_memory_over_time(records, axes[0][0], dataset=dataset, display=shown)
    panel_step_times(rows, axes[0][1], dataset=dataset, step_order=step_order,
                     display=shown)
    panel_peak_cost(rows, axes[1][0], dataset=dataset, display=shown)
    panel_scaling(rows, axes[1][1], predictions=predictions)
    _stamp(fig, records)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
