"""Per-process peak memory footprint, from the kernel (macOS only).

The headline memory figure. On macOS, a process's *physical footprint* is
the memory charged to it: its resident private pages plus its own pages the
compressor holds (plus IOKit and purgeable non-volatile memory). It is what
Activity Monitor shows as "Memory". Unlike peak RSS it does not drop when
the system compresses the process's memory, and unlike the system-wide
compressor counter it does not include other programs' pages. Clean
file-backed pages (a memory-mapped spill that the OS can simply drop) are
not charged, which is what "memory the analysis needs" should mean.

`proc_pid_rusage(pid, RUSAGE_INFO_V4)` reports the kernel's own lifetime
maximum of that footprint, so no peak can fall between two samples. It works
for any process of the same user, so the parent can read R subprocesses too.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

RUSAGE_INFO_V4 = 4
# struct rusage_info_v4: a 16-byte uuid, then uint64 fields. Offsets are
# counted in uint64 after the uuid.
_PHYS_FOOTPRINT = 7
_DISKIO_BYTES_READ = 16
_DISKIO_BYTES_WRITTEN = 17
_LIFETIME_MAX_PHYS_FOOTPRINT = 28

_libc = None
if sys.platform == "darwin":
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libc.proc_pid_rusage.restype = ctypes.c_int


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("u", ctypes.c_uint64 * 48)]


def rusage(pid: int) -> dict[str, int] | None:
    """Footprint and disk counters of ``pid``, or None where unavailable."""
    if _libc is None:
        return None
    buf = _RusageV4()
    if _libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        return None
    return {
        "phys_footprint": int(buf.u[_PHYS_FOOTPRINT]),
        "lifetime_max_phys_footprint": int(buf.u[_LIFETIME_MAX_PHYS_FOOTPRINT]),
        "diskio_bytes_read": int(buf.u[_DISKIO_BYTES_READ]),
        "diskio_bytes_written": int(buf.u[_DISKIO_BYTES_WRITTEN]),
    }
