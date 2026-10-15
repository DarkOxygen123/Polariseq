#!/usr/bin/env python
"""Peak memory of the neighbours step per phase, from a sweep of neighbors_memory.sbatch.

    python neighbors_phases.py RESULTS_DIR            # holds n<cells>/run.resources.csv and trace.log

For every size: the NN-Descent phases from its trace (rp-forest build, leaf pairs, random fill, each
iteration's candidates and join, extract) are placed on the monitor's clock, and the peak and mean of
anonymous memory (GB) and the mean cores in use are reported per phase, then the peak per cell.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def phases(trace: str):
    """[(label, seconds)] in order, from lines like `[nndescent] iter 3: join, 5 changed: 0.04s`."""
    out = []
    for line in trace.splitlines():
        m = re.match(r"\[nndescent\] (.*): ([0-9.]+)s$", line)
        if m and not m.group(1).startswith("total"):
            out.append((m.group(1), float(m.group(2))))
    return out


def main() -> int:
    root = Path(sys.argv[1])
    for d in sorted(root.glob("n*"), key=lambda p: int(p.name[1:])):
        n = int(d.name[1:])
        try:
            rows = list(csv.DictReader(open(d / "run.resources.csv")))
            trace = (d / "trace.log").read_text()
        except OSError:
            continue
        t = [float(r["t"]) for r in rows]
        anon = [float(r["anon_gb"]) for r in rows]
        rss = [float(r["rss_gb"]) for r in rows]
        cpu = [float(r["cpu_s"]) for r in rows]
        # the neighbours call starts after the embedding is read; its own trace begins at the first phase
        m = re.search(r"read in (\d+) s", trace)
        t_call = float(m.group(1)) if m else 0.0
        ph = phases(trace)
        total_trace = sum(s for _, s in ph)
        t_end_call = max(t)
        print(f"\n== {n:,} cells: peak anon {max(anon):.1f} GB, peak rss {max(rss):.1f} GB "
              f"({max(anon) * 1e9 / n:.0f} B/cell), call {t_end_call:.0f} s, nndescent trace {total_trace:.0f} s")
        # place phases on the clock, ending where NN-Descent ends: the last phase ends before the result
        # arrays are written and the fuzzy graph is built, so anchor on the first phase's start instead
        peak_i = max(range(len(anon)), key=lambda i: anon[i])
        start = None
        for i, (a, c) in enumerate(zip(anon, cpu)):
            if c > cpu[0] + 1.0 and t[i] > t_call:
                start = t[i]
                break
        start = start if start is not None else t_call
        clock = start
        print(f"   {'phase':34s} {'seconds':>8s} {'peak GB':>8s} {'mean GB':>8s} {'cores':>6s}")
        agg = {}
        for label, secs in ph:
            key = re.sub(r"iter \d+: ", "iter: ", label)
            lo, hi = clock, clock + secs
            idx = [i for i, x in enumerate(t) if lo <= x <= hi]
            if idx:
                a = [anon[i] for i in idx]
                cores = (cpu[idx[-1]] - cpu[idx[0]]) / max(1e-9, t[idx[-1]] - t[idx[0]])
                s, pk, mn, cs, cnt = agg.get(key, (0.0, 0.0, 0.0, 0.0, 0))
                agg[key] = (s + secs, max(pk, max(a)), mn + sum(a) / len(a), cs + cores, cnt + 1)
            clock = hi
        for key, (s, pk, mn, cs, cnt) in agg.items():
            print(f"   {key:34s} {s:8.0f} {pk:8.1f} {mn / cnt:8.1f} {cs / cnt:6.1f}")
        after = [i for i, x in enumerate(t) if x > clock]
        if after:
            print(f"   {'after NN-Descent (graph, fuzzy)':34s} {t_end_call - clock:8.0f} "
                  f"{max(anon[i] for i in after):8.1f}")
        print(f"   overall peak at {t[peak_i]:.0f} s into the run, phase clock started at {start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
