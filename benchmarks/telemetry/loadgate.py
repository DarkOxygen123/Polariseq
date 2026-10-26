"""Refuse to measure on a machine that is busy, and record why.

Measured on this M2 (2026-09-08, brain50k, polariseq-ram, identical code and
identical thread count): **22.3 s with load average 18 and 1.1 GB free, 10.8 s
once the machine was quieter.** A 2x swing from machine state alone. Wall-clock
numbers taken across that range are not comparable with each other, and
averaging them produces a figure that describes neither condition.

So a campaign that runs unattended for hours needs a gate rather than a note in
a protocol document. Every run records the machine state it started from, and a
run that starts outside the acceptance band is refused before it costs anything
rather than discarded afterwards.

Two ways to use it:

- ``wait_for_quiet()`` blocks until the machine settles, which is what an
  overnight campaign wants — a transient Spotlight index or a backup should
  delay the next run, not abort the campaign.
- ``check()`` returns the verdict without waiting, for a caller that would
  rather record a skip and move on.

The bands are deliberately not hard-coded to this machine. Free memory at rest
here is ~1.5 GB of 8 GiB, which would fail any threshold written for a server,
so :func:`calibrate` measures the idle state and derives the band from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import psutil


@dataclass(frozen=True)
class Band:
    """Acceptance limits for starting a run."""

    #: 1-minute load average must be at or below this. Deliberately loose:
    #: load1 lags and is contaminated by our own previous run. `max_cpu_busy`
    #: is the discriminating criterion; this one only catches a machine that
    #: is heavily and persistently loaded.
    max_load1: float = 6.0
    #: Bytes of available memory required before starting.
    min_available_bytes: int = 1_000_000_000
    #: What the machine actually had available at rest, from `calibrate`
    #: (0 when the band was not calibrated). A caller that raises
    #: `min_available_bytes` to what a run is known to need must not raise it
    #: past this: memory the machine never has free at rest will not appear
    #: by waiting, and the gate would stall for its whole timeout and then
    #: run anyway. It matters most on Windows, where SysMain preloading and
    #: background services hold several GB at rest — the 13.86 GiB laptop
    #: here idles around 30-45 per cent used. (`available` counts the standby
    #: list on Windows, so it is memory the system will hand over on demand,
    #: not memory that must first be freed by hand.)
    idle_available_bytes: int = 0
    #: Maximum share of total CPU busy in a short sample, 0-1.
    max_cpu_busy: float = 0.35

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MachineState:
    """What the machine looked like at one instant."""

    load1: float
    load5: float
    available_bytes: int
    total_bytes: int
    cpu_busy: float
    swap_used_bytes: int
    cpu_freq_mhz: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample(band: Band, *, cpu_window_s: float = 0.5) -> MachineState:
    """Sample machine state once and judge it against `band`."""
    load1, load5, _ = psutil.getloadavg()
    vm = psutil.virtual_memory()
    # `interval` blocks for the window and returns busy share over it, which
    # is what we want — an instantaneous reading is mostly noise.
    cpu_busy = psutil.cpu_percent(interval=cpu_window_s) / 100.0
    try:
        swap_used = psutil.swap_memory().used
    except Exception:
        swap_used = 0
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = float(f.current) if f else None
    except Exception:
        freq = None

    reasons: list[str] = []
    if load1 > band.max_load1:
        reasons.append(f"load1 {load1:.1f} > {band.max_load1:.1f}")
    if vm.available < band.min_available_bytes:
        reasons.append(
            f"available {vm.available / 1e9:.2f} GB < "
            f"{band.min_available_bytes / 1e9:.2f} GB"
        )
    if cpu_busy > band.max_cpu_busy:
        reasons.append(f"cpu busy {cpu_busy:.0%} > {band.max_cpu_busy:.0%}")

    return MachineState(
        load1=round(load1, 2),
        load5=round(load5, 2),
        available_bytes=int(vm.available),
        total_bytes=int(vm.total),
        cpu_busy=round(cpu_busy, 3),
        swap_used_bytes=int(swap_used),
        cpu_freq_mhz=round(freq, 1) if freq else None,
        reasons=reasons,
    )


def check(band: Band | None = None) -> MachineState:
    """One-shot verdict. `state.ok` is False when the machine is too busy."""
    return sample(band or Band())


def wait_for_quiet(
    band: Band | None = None,
    *,
    timeout_s: float = 1800.0,
    poll_s: float = 20.0,
    verbose: bool = True,
) -> MachineState:
    """Block until the machine is quiet enough, or until `timeout_s`.

    Returns the final state either way. A caller that gets back a state with
    ``ok is False`` has waited the full timeout and should record a skip
    rather than measure through the interference.
    """
    band = band or Band()
    deadline = time.monotonic() + timeout_s
    state = sample(band)
    announced = False
    while not state.ok and time.monotonic() < deadline:
        if verbose and not announced:
            print(
                f"    [loadgate] waiting for a quiet machine: "
                f"{'; '.join(state.reasons)}",
                flush=True,
            )
            announced = True
        time.sleep(poll_s)
        state = sample(band)
    if verbose and announced and state.ok:
        print("    [loadgate] machine settled, starting", flush=True)
    return state


def calibrate(
    *, samples: int = 6, gap_s: float = 5.0, verbose: bool = True
) -> Band:
    """Derive an acceptance band from this machine's current idle behaviour.

    A threshold written for one machine is wrong on another. Free memory at
    rest on the 8 GiB M2 is around 1.5 GB, so a fixed 4 GB minimum would never
    admit a run, while on a 64 GB workstation it would admit anything.

    The band allows some headroom above observed idle, because idle is not the
    quietest the machine ever is and a gate that never opens is useless.
    """
    loads, avails, busies = [], [], []
    for _ in range(samples):
        s = sample(Band(max_load1=1e9, min_available_bytes=0, max_cpu_busy=2.0))
        loads.append(s.load1)
        avails.append(s.available_bytes)
        busies.append(s.cpu_busy)
        time.sleep(gap_s)

    med_load = sorted(loads)[len(loads) // 2]
    med_avail = sorted(avails)[len(avails) // 2]
    med_busy = sorted(busies)[len(busies) // 2]

    band = Band(
        idle_available_bytes=int(med_avail),
        # Load average is a *lagging* 1-minute mean and cannot tell our own
        # previous run from someone else's work. Measured 2026-09-09: a
        # 4-thread 15 s run raises load1 by ~1.0, which then decays over more
        # than a minute, so a band calibrated from idle stalls the campaign
        # between every run. The limit therefore carries headroom for our own
        # decay, and `cpu_busy` — sampled between runs, when nothing of ours
        # is executing — is the criterion that actually detects interference.
        max_load1=round(max(6.0, med_load + 4.0), 2),
        # Require most of the memory that was free at rest, so a run does not
        # start into a machine that has since filled up.
        min_available_bytes=int(max(500_000_000, med_avail * 0.7)),
        max_cpu_busy=round(min(0.6, max(0.25, med_busy + 0.20)), 3),
    )
    if verbose:
        print(
            f"[loadgate] calibrated from idle: load1<={band.max_load1}, "
            f"available>={band.min_available_bytes / 1e9:.2f} GB "
            f"(of {band.idle_available_bytes / 1e9:.2f} GB free at rest), "
            f"cpu_busy<={band.max_cpu_busy:.0%}"
        )
    return band


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import json
    import sys

    if "--calibrate" in sys.argv:
        print(json.dumps(calibrate().as_dict(), indent=2))
    else:
        st = check()
        print(json.dumps(st.as_dict(), indent=2))
        print("VERDICT:", "ok" if st.ok else "too busy — " + "; ".join(st.reasons))
        sys.exit(0 if st.ok else 1)
