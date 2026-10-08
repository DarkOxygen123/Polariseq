"""The measured process. Run as ``python -m telemetry.child CONFIG.json``.

Every arm runs in a fresh subprocess. That is not fastidiousness: a peak
memory figure is a process-wide high-water mark, so two arms sharing an
interpreter would each inherit the other's worst moment, and scanpy's import
graph would be charged to whichever arm happened to run second. A fresh
process also means an OOM kill takes down the run being measured and nothing
else, and the parent still gets to record it as a result — "did not fit" is
data, not a missing row.

The child writes its half of the record to ``config["child_out"]`` on the way
out, including when the arm raises. The parent merges it with the half it
measured from outside.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[1]).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from telemetry.arms import ARMS
    from telemetry.record import Params
    from telemetry.footprint import rusage
    from telemetry.sampler import MemorySampler, Timeline

    params = Params(**cfg["params"])
    arm = ARMS[cfg["arm"]]
    artifacts = Path(cfg["artifacts_dir"])
    out_path = Path(cfg["child_out"])

    timeline = Timeline()
    sampler = MemorySampler(interval_s=cfg.get("interval_s", 0.02))
    sampler.start(origin=timeline.origin)

    payload: dict[str, object] = {"status": "ok", "note": "", "outputs": {}}
    t0 = time.perf_counter()
    try:
        # Baseline is taken *after* the arm's imports, which happen inside
        # `run`. The arms import their library first thing, so the first
        # sample inside `load` already includes it; `mark_baseline` here
        # captures the interpreter-only floor, and the load step's own start
        # value captures the post-import floor. Both are reported.
        sampler.mark_baseline()
        payload["outputs"] = arm.run(cfg["path"], params, timeline, artifacts)
    except MemoryError:
        payload["status"] = "oom"
        payload["note"] = "MemoryError raised in-process"
    except BaseException as exc:  # noqa: BLE001 - record everything, then re-raise nothing
        payload["status"] = "failed"
        payload["note"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-4000:]
    finally:
        wall = time.perf_counter() - t0
        sampler.stop()
        payload.update(
            {
                # The kernel's lifetime peak footprint of this process,
                # read last so nothing after it can raise the peak.
                "footprint": rusage(os.getpid()) or {},
                "wall_s": round(wall, 4),
                "steps": timeline.as_list(),
                "telemetry": sampler.as_dict(),
                "arm_tuning_notes": arm.tuning_notes,
                "arm_label": arm.label,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
