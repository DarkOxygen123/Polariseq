"""Child-side instrumentation: a memory sampler and a step timeline.

Both live *inside* the process being measured, because the number that
matters — USS, the memory that would be returned to the OS if this process
exited — is only readable for oneself on macOS (``task_for_pid`` denies it
across processes; measured, see the handbook chapter 09).

The sampler is a plain Python thread, and that has a consequence which this
module refuses to hide: **a thread cannot run while another thread holds the
GIL.** Polariseq's Rust kernels currently hold it for their whole duration,
so during a 0.7 s PCA the sampler observed 1 sample where 36 were due. A
sampler that is blind exactly during the expensive step would draw a
flattering flat line for whichever arm holds the GIL hardest, which is the
opposite of what this harness is for.

So every record carries :meth:`MemorySampler.gaps`, and the analysis refuses
to plot a series whose blind fraction exceeds a threshold. The parent-side
sampler in ``runner.py`` covers the same interval with RSS from outside the
process, where no GIL applies, and the two series cross-validate each other.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

#: Sampling period. 0.02 s costs ~0.08 ms per sample (measured on an M2:
#: `memory_full_info()` does not get slower as the process grows), i.e. well
#: under 1 % overhead, while still resolving a step that lasts a tenth of a
#: second.
DEFAULT_INTERVAL_S = 0.02

#: Minimum period between USS reads. On Windows psutil computes USS by
#: walking the process's entire working set (`QueryWorkingSet`), so its cost
#: grows with the process: at 50 reads a second on a multi-GB run it would
#: slow the very process being measured, and a Python-heavy arm more than a
#: native one. There USS is read twice a second and carried forward between
#: reads, while RSS keeps the full rate. macOS and Linux read it every tick.
USS_INTERVAL_S = 0.5 if sys.platform == "win32" else 0.0

#: A run whose sampler was blind for more than this fraction of the wall
#: clock has not been measured, it has been guessed at. `analyze` marks such
#: records `degraded` rather than quietly plotting them.
MAX_BLIND_FRACTION = 0.05


@dataclass
class Step:
    """One named stage of a pipeline, timed on the shared clock."""

    name: str
    t_begin: float
    t_end: float | None = None

    @property
    def wall_s(self) -> float:
        return (self.t_end - self.t_begin) if self.t_end is not None else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "t_begin": round(self.t_begin, 4),
            "t_end": round(self.t_end, 4) if self.t_end is not None else None,
            "wall_s": round(self.wall_s, 4),
        }


class _StepContext:
    def __init__(self, timeline: Timeline, step: Step) -> None:
        self._timeline = timeline
        self.step = step

    def __enter__(self) -> Step:
        return self.step

    def __exit__(self, *exc: object) -> bool:
        self.step.t_end = self._timeline.now()
        return False


@dataclass
class Timeline:
    """Named steps on the same clock as the memory samples.

    The clock is ``time.perf_counter()`` offset so that ``t = 0`` is the
    moment sampling started. Sharing one origin is what lets the analysis
    attribute a memory plateau to the step that caused it.
    """

    origin: float = field(default_factory=time.perf_counter)
    steps: list[Step] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter() - self.origin

    def step(self, name: str) -> _StepContext:
        """Time a block: ``with timeline.step("pca"): ...``"""
        s = Step(name=name, t_begin=self.now())
        self.steps.append(s)
        return _StepContext(self, s)

    def as_list(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.steps]


class MemorySampler:
    """Samples this process's memory on a background thread.

    Captures, per sample:

    - ``rss``  — resident set size. Includes shared pages and, critically,
      memory-mapped file pages that are really page cache. RSS *over*states
      what an out-of-core implementation costs.
    - ``uss``  — unique set size: pages that would be freed if this process
      died. This is the honest "what does this tool cost me" number, and it
      *under*states cost for a workload whose speed depends on page cache
      that some other process is keeping warm.
    - ``pfaults`` / ``pageins`` — cumulative fault counters (macOS reports
      both; Linux exposes them via the same psutil fields where available).
      ``pageins`` rising is the signature of actually touching the disk, and
      is how the mmap-vs-block-I/O argument is settled with data rather than
      opinion.

    Reporting both RSS and USS is a requirement, not a courtesy: quoting
    whichever one flatters the arm you like is the standard way benchmarks
    of this kind go wrong.
    """

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.origin: float = 0.0
        self.t0_unix: float = 0.0
        self.t: list[float] = []
        self.rss: list[int] = []
        self.uss: list[int] = []
        self.pfaults: list[int] = []
        self.pageins: list[int] = []
        #: Filled by :meth:`mark_baseline`, after imports and before work.
        self.baseline: dict[str, int] = {}
        self._uss_available = True
        self.uss_interval_s = USS_INTERVAL_S
        self._uss_last = 0
        self._uss_read_at = -1e9

    # -- lifecycle ---------------------------------------------------------

    def start(self, origin: float | None = None) -> MemorySampler:
        """Begin sampling. Pass a :class:`Timeline`'s origin to share a clock."""
        self.origin = time.perf_counter() if origin is None else origin
        self.t0_unix = time.time()
        self._thread = threading.Thread(
            target=self._loop, name="polariseq-telemetry", daemon=True
        )
        self._thread.start()
        # Do not return before at least one sample exists, so that a step
        # beginning immediately has something to sit on.
        deadline = time.perf_counter() + 1.0
        while not self.t and time.perf_counter() < deadline:
            time.sleep(0.001)
        return self

    def stop(self) -> MemorySampler:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def __enter__(self) -> MemorySampler:
        return self.start()

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False

    # -- sampling ----------------------------------------------------------

    def _sample_once(self, *, force_uss: bool = False) -> None:
        try:
            now = time.perf_counter()
            if self._uss_available and (
                force_uss or now - self._uss_read_at >= self.uss_interval_s
            ):
                info = self._proc.memory_full_info()
                uss = self._uss_last = int(getattr(info, "uss", 0))
                self._uss_read_at = now
            else:
                info = self._proc.memory_info()
                uss = self._uss_last if self._uss_available else 0
        except psutil.AccessDenied:
            # Some sandboxes deny the page-table walk that USS needs. Degrade
            # to RSS-only rather than losing the series, and let the record
            # say so via `uss_available`.
            self._uss_available = False
            return
        except (psutil.NoSuchProcess, OSError):
            return
        self.t.append(time.perf_counter() - self.origin)
        self.rss.append(int(info.rss))
        self.uss.append(uss)
        self.pfaults.append(int(getattr(info, "pfaults", 0)))
        self.pageins.append(int(getattr(info, "pageins", 0)))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def mark_baseline(self) -> None:
        """Record the post-import, pre-work footprint.

        Every arm pays an import cost — scanpy's stack and Polariseq's
        compiled extension are not the same size — and a comparison that
        silently folds that into the pipeline is measuring the wrong thing.
        Recording it separately lets the analysis show both the absolute
        footprint and the work-attributable delta above baseline.
        """
        self._sample_once(force_uss=True)
        if self.t:
            self.baseline = {
                "rss": self.rss[-1],
                "uss": self.uss[-1],
                "t": round(self.t[-1], 4),
            }

    # -- results -----------------------------------------------------------

    def gaps(self) -> dict[str, Any]:
        """How well the sampler actually kept up.

        ``blind_fraction`` is the share of wall time spent in gaps longer
        than twice the requested interval — i.e. time during which this
        process could have allocated and freed a gigabyte unobserved.
        """
        if len(self.t) < 2:
            return {
                "n": len(self.t),
                "max_s": None,
                "p99_s": None,
                "median_s": None,
                "blind_fraction": 1.0,
                "uss_available": self._uss_available,
            }
        deltas = [b - a for a, b in zip(self.t, self.t[1:], strict=False)]
        span = self.t[-1] - self.t[0]
        threshold = 2.0 * self.interval_s
        blind = sum(d for d in deltas if d > threshold)
        ordered = sorted(deltas)
        p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        return {
            "n": len(self.t),
            "max_s": round(max(deltas), 4),
            "p99_s": round(p99, 4),
            "median_s": round(ordered[len(ordered) // 2], 4),
            "blind_fraction": round(blind / span, 4) if span > 0 else 1.0,
            "uss_available": self._uss_available,
        }

    def peaks(self) -> dict[str, int]:
        return {
            "rss": max(self.rss) if self.rss else 0,
            "uss": max(self.uss) if self.uss else 0,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_s": self.interval_s,
            "uss_interval_s": self.uss_interval_s,
            "t0_unix": self.t0_unix,
            "samples": {
                "t": [round(x, 4) for x in self.t],
                "rss": self.rss,
                "uss": self.uss,
                "pfaults": self.pfaults,
                "pageins": self.pageins,
            },
            "gaps": self.gaps(),
            "peaks": self.peaks(),
            "baseline": self.baseline,
        }
