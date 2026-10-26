"""The worker pool and the budget's thread count.

The pool is built on first parallel use and cannot be resized, so each case
runs in a fresh process. Work that runs before any budget is read (an object
made from an array) sizes the pool by itself: the budget then takes the
pool's count without a warning, since nobody asked for another. A count
asked for with ``set_budget(threads=...)`` that cannot take effect still warns.
"""

from __future__ import annotations

import subprocess
import sys

START = """
import warnings
import numpy as np
import polariseq as ps
warnings.simplefilter("error", RuntimeWarning)  # any thread warning fails the run
a = ps.AnnData(X=np.random.default_rng(0).poisson(1, (300, 50)).astype(np.float32))
ps.pp.normalize_total(a)
ps.pp.log1p(a)
ps.pp.pca(a, 10)   # parallel work before any budget was read
"""


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", START + code], capture_output=True, text=True,
                          timeout=300)


def test_a_detected_budget_takes_the_pool_it_finds_without_a_warning() -> None:
    r = run("""
ps.pp.neighbors(a)
assert ps.budget().threads == ps._thread_count(), (ps.budget().threads, ps._thread_count())
ps.set_budget(ram_gb=2)   # no thread count asked for: still nothing to warn about
assert ps.budget().threads == ps._thread_count()
print("ok")
""")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_a_thread_count_asked_for_too_late_still_warns() -> None:
    r = run("""
warnings.simplefilter("default", RuntimeWarning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ps.set_budget(threads=ps._thread_count() + 1)
assert any("could not set the worker pool" in str(x.message) for x in w), [str(x.message) for x in w]
assert ps.budget().threads == ps._thread_count()   # the budget says what is really running
print("ok")
""")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
