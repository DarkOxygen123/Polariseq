"""What an arm is, and what it owes the comparison.

An *arm* is one implementation of one fixed pipeline. The pipeline is fixed
so that the only thing varying between arms is the implementation:

    load -> qc -> filter_cells -> normalize -> log1p -> hvg
         -> pca -> neighbors -> leiden -> [umap]

Two obligations make this a comparison rather than a race:

1. **Same step names.** The analysis aligns memory curves by step. An arm
   that fuses normalize and log1p still emits both markers (the second may
   be near-zero) so the panels line up and the fusion is *visible* instead
   of being hidden in a differently-named bar.

2. **Same artifacts.** Every arm writes its PCA embedding, its cluster
   labels, its HVG gene names, and the cell barcodes they correspond to.
   Without these, a speed win is indistinguishable from a correctness loss,
   and the harness would be measuring how fast each library can be wrong.
   ``equivalence.py`` consumes exactly these files.

Arms must not tune the *pipeline* (fewer PCs, looser convergence) to win.
They may tune the *configuration* of their own library, and every such
choice is declared in ``tuning_notes`` so a reviewer can object to it. The
``scanpy-tuned`` arm exists precisely so that the honest question — "is
Polariseq faster than scanpy, or just faster than scanpy's defaults?" — is
answered in the results rather than dodged.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

#: The canonical step order. Arms emit these names; extra names are allowed
#: (e.g. Polariseq's `spill`), but these must all appear so panels align.
CANONICAL_STEPS = (
    "import",
    "load",
    "qc",
    "filter_cells",
    "normalize",
    "log1p",
    "hvg",
    "pca",
    "neighbors",
    "leiden",
    "umap",
)


class Arm(Protocol):
    """One implementation of the fixed pipeline."""

    name: str
    label: str
    tuning_notes: str

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        """Execute the pipeline, emitting a ``timeline.step`` per stage.

        Returns a summary dict (cell/gene counts, cluster count, and the
        artifact paths written under ``artifacts``).
        """
        ...


def write_artifacts(
    directory: Path,
    *,
    embedding: np.ndarray | None,
    labels: np.ndarray | None,
    hvg_names: list[str] | None,
    obs_names: list[str] | None,
    umap: np.ndarray | None = None,
) -> dict[str, str]:
    """Persist the outputs the equivalence check needs.

    Stored as ``.npy``/newline-delimited text rather than inside the JSON
    record: a 100k x 50 embedding is 20 MB, which belongs beside the record,
    not inside it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    if embedding is not None:
        p = directory / "pca.npy"
        np.save(p, np.asarray(embedding, dtype=np.float32))
        written["pca"] = str(p)
    if umap is not None:
        p = directory / "umap.npy"
        np.save(p, np.asarray(umap, dtype=np.float32))
        written["umap"] = str(p)
    if labels is not None:
        p = directory / "labels.npy"
        np.save(p, np.asarray(labels, dtype=np.int32))
        written["labels"] = str(p)
    if hvg_names is not None:
        p = directory / "hvg.txt"
        p.write_text("\n".join(map(str, hvg_names)))
        written["hvg"] = str(p)
    if obs_names is not None:
        p = directory / "obs_names.txt"
        p.write_text("\n".join(map(str, obs_names)))
        written["obs_names"] = str(p)
    return written


def dataset_facts(path: str) -> dict[str, int]:
    """Shape and nnz from the file's metadata, without loading the matrix.

    Read in the parent before any arm starts, so that all arms are recorded
    against identical dataset facts.
    """
    import h5py

    facts = {"n_cells": 0, "n_genes": 0, "nnz": 0, "file_bytes": 0}
    with contextlib.suppress(OSError):
        facts["file_bytes"] = Path(path).stat().st_size
    try:
        with h5py.File(path, "r") as f:
            x = f["X"]
            if isinstance(x, h5py.Group):  # sparse
                shape = tuple(x.attrs.get("shape", (0, 0)))
                facts["n_cells"], facts["n_genes"] = int(shape[0]), int(shape[1])
                facts["nnz"] = int(x["data"].shape[0])
            else:  # dense
                facts["n_cells"], facts["n_genes"] = int(x.shape[0]), int(x.shape[1])
                facts["nnz"] = facts["n_cells"] * facts["n_genes"]
    except (OSError, KeyError, ValueError):
        pass
    return facts
