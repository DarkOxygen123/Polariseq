#!/usr/bin/env python
"""Run one command and measure it: memory, CPU and disk sampled twice a second, tagged by the
command's own STEP lines (see common.py).

    python monitor.py --out RUN_DIR --name run [--watch DIR] -- COMMAND ARGS...

Writes into RUN_DIR:

    <name>.resources.csv   one row per sample: seconds since start, resident memory of the whole
                           process tree (GB), its anonymous part (heap, not file pages), cumulative
                           CPU seconds, processes alive, scratch quota used (GB, every 30 s)
    <name>.stdout.log, <name>.stderr.log
    <name>.monitor.json    exit code, wall time, overall peaks, and per step: seconds, peak resident
                           and anonymous memory, mean cores busy; the command's own JSON summary
                           (last stdout line); sizes of the --watch folders at the end

Resident memory counts file pages a process has mapped (memory-mapped stores do this). The
anonymous part is what the process allocated itself; both are reported, the resident one is the
headline (it is what the scheduler's limit sees on most clusters). The exit code is the command's.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

GB = 1e9


def rss_anon(pid: int) -> float:
    """RssAnon of a process in bytes (Linux); NaN elsewhere."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return float("nan")


def quota_gb() -> float | None:
    """Scratch space used by this user (Lustre quota), or None off Lustre."""
    try:
        out = subprocess.run(["lfs", "quota", "-q", "-u", os.environ.get("USER", ""), "/scratch"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return round(int(out[1].rstrip("*")) * 1024 / GB, 2)
    except Exception:
        return None


def du_gb(path: str) -> float | None:
    try:
        out = subprocess.run(["du", "-sk", path], capture_output=True, text=True).stdout.split()
        return round(int(out[0]) * 1024 / GB, 2)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--watch", action="append", default=[], help="folder whose size is reported at the end")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / a.name

    import signal
    stopped = {"by": None}

    def on_term(signum, frame):   # Slurm's time limit or scancel: record what was measured, then stop
        stopped["by"] = signal.Signals(signum).name
        try:
            for p in psutil.Process(proc.pid).children(recursive=True) + [psutil.Process(proc.pid)]:
                p.terminate()
        except (psutil.Error, NameError):
            pass

    signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    load0 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None   # other jobs already on the node
    q0 = quota_gb()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    root = psutil.Process(proc.pid)
    steps: list[tuple[str, float, float]] = []
    last_stdout = [""]

    def pump(stream, log_path, echo, parse):
        with open(log_path, "w") as log:
            for line in stream:
                log.write(line)
                log.flush()
                echo.write(line)
                echo.flush()
                if parse and line.startswith("STEP "):
                    try:
                        _, name, b, e = line.split()
                        steps.append((name, float(b), float(e)))
                    except ValueError:
                        pass
                if not parse and line.strip():
                    last_stdout[0] = line.strip()

    threads = [threading.Thread(target=pump, args=(proc.stdout, f"{stem}.stdout.log", sys.stdout, False)),
               threading.Thread(target=pump, args=(proc.stderr, f"{stem}.stderr.log", sys.stderr, True))]
    for t in threads:
        t.start()

    cpu_seen: dict[int, float] = {}
    samples: list[tuple] = []   # t, rss, anon, cpu, nproc, quota
    quota_last, quota_t = q0, t0
    with open(f"{stem}.resources.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "rss_gb", "anon_gb", "cpu_s", "n_procs", "quota_gb"])
        while True:
            now = time.time()
            rss = anon = 0.0
            n = 0
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = []
            for p in procs:
                try:
                    with p.oneshot():
                        rss += p.memory_info().rss
                        ct = p.cpu_times()
                    cpu_seen[p.pid] = max(cpu_seen.get(p.pid, 0.0), ct.user + ct.system)
                    anon += rss_anon(p.pid)
                    n += 1
                except psutil.Error:
                    pass
            if now - quota_t >= 30:
                quota_last, quota_t = quota_gb(), now
            row = (round(now - t0, 2), round(rss / GB, 3), round(anon / GB, 3),
                   round(sum(cpu_seen.values()), 1), n, quota_last)
            samples.append((now, rss, anon, sum(cpu_seen.values()), n, quota_last))
            w.writerow(row)
            fh.flush()
            if proc.poll() is not None:
                break
            time.sleep(a.interval)
    for t in threads:
        t.join()
    t1 = time.time()

    def window(b: float, e: float) -> dict:
        inside = [s for s in samples if b <= s[0] <= e]
        before = [s for s in samples if s[0] <= b] or samples[:1]
        after = [s for s in samples if s[0] >= e] or samples[-1:]
        pool = inside or [before[-1], after[0]]
        cpu = after[0][3] - before[-1][3]
        span = max(e - b, 1e-9)
        return {"seconds": round(e - b, 2),
                "peak_rss_gb": round(max(s[1] for s in pool) / GB, 2),
                "peak_anon_gb": round(max(s[2] for s in pool) / GB, 2),
                "mean_cores": round(cpu / max(after[0][0] - before[-1][0], span), 2),
                "samples": len(inside)}

    driver = None
    try:
        driver = json.loads(last_stdout[0])
    except (ValueError, TypeError):
        pass
    quotas = [s[5] for s in samples if s[5] is not None]
    res = {
        "name": a.name, "cmd": cmd, "exit_code": proc.returncode, "stopped_by": stopped["by"],
        "host": socket.gethostname(), "load_at_start": load0, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "slurm_mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "wall_s": round(t1 - t0, 2),
        "peak_rss_gb": round(max(s[1] for s in samples) / GB, 2),
        "peak_anon_gb": round(max(s[2] for s in samples) / GB, 2),
        "cpu_s": round(samples[-1][3], 1),
        "mean_cores": round(samples[-1][3] / max(t1 - t0, 1e-9), 2),
        "quota_gb": {"start": q0, "peak": max(quotas) if quotas else None, "end": quota_gb()},
        "watch_gb": {w: du_gb(w) for w in a.watch},
        "steps": {name: window(b, e) for name, b, e in steps},
        "step_order": [s[0] for s in steps],
        "driver": driver,
    }
    Path(f"{stem}.monitor.json").write_text(json.dumps(res, indent=1))
    print(f"[monitor] {a.name}: exit {proc.returncode}, {res['wall_s'] / 60:.1f} min, "
          f"peak {res['peak_rss_gb']} GB", file=sys.stderr, flush=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
