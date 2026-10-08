"""The arm registry. Adding an arm means adding it here and nowhere else."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import CANONICAL_STEPS, Arm, dataset_facts, write_artifacts
from .polariseq_arm import PolariseqRam, PolariseqSpill
from .scanpy_arm import ScanpyDefault, ScanpyTuned, ScanpyTunedInt64, ScanpyTunedSeed1

_arms: list[Arm] = [
    PolariseqRam(), PolariseqSpill(),
    ScanpyDefault(), ScanpyTuned(), ScanpyTunedSeed1(), ScanpyTunedInt64(),
]

# The Seurat arms need R on PATH. Registering them unconditionally would make
# `--arms` listings advertise something that cannot run, and would break the
# Python-only campaign on a machine without R, so they are added only when
# Rscript is actually present.
if shutil.which("Rscript"):
    from .seurat_arm import BPCellsNative, SeuratBPCells, SeuratDefault, SeuratTuned

    _arms += [SeuratDefault(), SeuratTuned(), SeuratBPCells(), BPCellsNative()]

# Scarf needs its own interpreter (see scarf_arm.py); registered when present.
if (Path(__file__).resolve().parents[3] / ".venv-scarf" / "bin" / "python").exists():
    from .scarf_arm import ScarfArm

    _arms += [ScarfArm()]

ARMS: dict[str, Arm] = {a.name: a for a in _arms}  # type: ignore[misc]

#: The default comparison set: ours in RAM against both scanpy configurations.
#: The spill arm is opt-in because it is a different claim (see its docstring)
#: and because on a dataset that fits in RAM it only demonstrates the cost of
#: choosing not to use it.
DEFAULT_ARMS = ("polariseq-ram", "scanpy-default", "scanpy-tuned")

#: The seed-to-seed floor arm is opt-in like the spill arm: it duplicates
#: scanpy-tuned's timings, so it costs wall time and answers exactly one
#: question — what does the same implementation score against itself at a
#: second seed? Only useful on a with-UMAP campaign (the floor is a kNN
#: overlap of layouts).

__all__ = [
    "ARMS", "DEFAULT_ARMS", "Arm", "CANONICAL_STEPS",
    "dataset_facts", "write_artifacts",
]
