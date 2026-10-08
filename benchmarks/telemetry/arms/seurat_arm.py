"""Seurat arms: the R side of the comparison, driven as a subprocess.

Three things make this arm different from the Python ones, and each is a
decision a reviewer could object to, so each is stated here.

**Memory.** The in-process sampler is Python and cannot see into an R child.
Peak memory for these arms comes from the parent sampler, which sums the whole
process tree (`ParentSampler`, fixed 2026-09-09 — before that it watched only
the direct child and would have recorded a Seurat run as costing nothing).
USS is therefore unavailable for Seurat and the record says so rather than
reporting a zero.

**Format conversion is not timed.** Seurat cannot read `.h5ad`. Converting it
to a `dgCMatrix` (and, for the BPCells arm, to a bit-packed directory) happens
*before* the timed region, exactly as the Python arms read a file already
converted for them. Charging Seurat for a conversion the others do not pay
would measure the file format, not the implementation.

**Step markers cross the process boundary.** The R script prints
`STEP <name> <begin> <end>` on a clock whose origin is its own start. Those are
replayed into the Python `Timeline` after the run, offset to the subprocess
launch, so the panels align with every other arm.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import write_artifacts

R_SCRIPT = Path(__file__).parent / "R" / "seurat_pipeline.R"
R_BPCELLS_NATIVE = Path(__file__).parent / "R" / "bpcells_native.R"
R_BPCELLS_FROM_H5AD = Path(__file__).parent / "R" / "bpcells_from_h5ad.R"


def _run_r(script: Path, cfg_path: Path, env: dict[str, str]) -> tuple[int, str, str, int]:
    """Run an R script; see :func:`run_tracked`."""
    return run_tracked(["Rscript", "--vanilla", str(script), str(cfg_path)], env)


def run_tracked(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str, int]:
    """Run ``cmd``; return (returncode, stdout, stderr, peak footprint).

    The peak is the subprocess's own lifetime maximum footprint, polled from
    the kernel until it exits (macOS; 0 elsewhere). The measured Python
    process around it also holds the untimed input conversion, so its own
    footprint is not the arm's: the record uses this figure instead.
    """
    from ..footprint import rusage

    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=env)
        peak = 0
        while proc.poll() is None:
            r = rusage(proc.pid)
            if r is not None:
                peak = max(peak, r["lifetime_max_phys_footprint"])
            time.sleep(0.05)
        out.seek(0)
        err.seek(0)
        return proc.returncode, out.read(), err.read(), peak


def _h5ad_to_rds(path: str, out: Path) -> None:
    """Convert the AnnData matrix to a dgCMatrix RDS, outside the timed region."""
    import anndata as ad
    import scipy.sparse as sp

    adata = ad.read_h5ad(path)
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # R's dgCMatrix is CSC and genes-by-cells, the transpose of AnnData's
    # cells-by-genes. Written as raw little-endian binary rather than .npy or
    # Matrix Market: R reads binary directly with readBin, whereas .npy needs
    # a parser and .mtx would be a ~1.5 GB text file for pbmc20k alone.
    Xt = sp.csc_matrix(X.T)
    # R's dgCMatrix validity requires row indices ascending within each
    # column. scipy does not guarantee that after a transpose, and the
    # violation is silent until R rejects the object with "'i' slot is not
    # increasing within columns" — which pbmc3k happened to dodge and every
    # larger matrix did not.
    Xt.sort_indices()
    Xt.sum_duplicates()
    Xt.data.astype("<f8").tofile(str(out) + ".data.bin")
    Xt.indices.astype("<i4").tofile(str(out) + ".indices.bin")
    Xt.indptr.astype("<i4").tofile(str(out) + ".indptr.bin")
    # Seurat rewrites '_' to '-' in feature names on object creation
    # ("Feature names cannot have underscores"), which would make the HVG
    # Jaccard against scanpy fail for a naming reason rather than a selection
    # one. Applying the same substitution here means both tools see identical
    # names and the comparison measures gene choice, not string handling.
    genes = [str(g).replace("_", "-") for g in adata.var_names]
    Path(str(out) + ".dims.json").write_text(json.dumps({
        "nrow": int(Xt.shape[0]), "ncol": int(Xt.shape[1]),
        "nnz": int(Xt.nnz),
        "genes": genes,
        "cells": [str(c) for c in adata.obs_names],
    }))


def _bpcells_readable(path: str, out: Path) -> None:
    """A small h5ad that BPCells' AnnData reader accepts, without a copy.

    BPCells 0.3.1 reads ``obs/_index`` and ``var/_index`` as plain string
    datasets; anndata 0.13 writes indices as nullable-string groups (and may
    name them otherwise), which BPCells rejects. This file links ``X`` to
    the original matrix (an HDF5 external link: no data copied) and stores
    the two indices as fixed-length strings.
    """
    import anndata as ad
    import h5py

    with h5py.File(path, "r") as f:
        obs = ad.io.read_elem(f["obs"]).index.astype(str).to_numpy()
        var = ad.io.read_elem(f["var"]).index.astype(str).to_numpy()
    with h5py.File(out, "w") as g:
        g["X"] = h5py.ExternalLink(str(Path(path).resolve()), "/X")
        g.create_dataset("obs/_index", data=obs.astype("S"))
        g.create_dataset("var/_index", data=var.astype("S"))
        for grp in ("obs", "var"):
            g[grp].attrs["_index"] = "_index"
            g[grp].attrs["encoding-type"] = "dataframe"
            g[grp].attrs["encoding-version"] = "0.2.0"


def _replay_steps(stderr: str, timeline: Any, t_offset: float) -> None:
    """Fold the R script's STEP markers into the Python timeline."""
    from ..sampler import Step

    for line in stderr.splitlines():
        if not line.startswith("STEP "):
            continue
        try:
            _, name, b, e = line.split()
            timeline.steps.append(
                Step(name=name, t_begin=t_offset + float(b),
                     t_end=t_offset + float(e))
            )
        except ValueError:
            continue


def _csv_to_npy(artifacts: Path) -> dict[str, Any]:
    """Convert what R wrote into what `equivalence.py` reads."""
    out: dict[str, Any] = {"embedding": None, "labels": None,
                           "hvg_names": None, "obs_names": None, "umap": None}
    pca = artifacts / "pca.csv"
    if pca.exists():
        out["embedding"] = np.loadtxt(pca, delimiter=",", dtype=np.float32)
    lab = artifacts / "labels.csv"
    if lab.exists():
        out["labels"] = np.loadtxt(lab, delimiter=",", dtype=np.int32).reshape(-1)
    hvg = artifacts / "hvg.txt"
    if hvg.exists():
        out["hvg_names"] = hvg.read_text().split()
    obs = artifacts / "obs_names.txt"
    if obs.exists():
        out["obs_names"] = obs.read_text().split()
    um = artifacts / "umap.csv"
    if um.exists():
        out["umap"] = np.loadtxt(um, delimiter=",", dtype=np.float32)
    return out


class _SeuratBase:
    use_bpcells = False
    r_script = R_SCRIPT

    def run(self, path: str, params: Any, timeline: Any,
            artifacts: Path) -> dict[str, Any]:
        """Convert the input, run the R pipeline, and always clean up after it.

        The conversion writes nnz*8 bytes for the data and nnz*4 for the
        indices, so one brain200k run leaves 4.8 GB behind. `mkdtemp` had no
        matching cleanup and every run leaked its directory: measured
        2026-09-10, 140 abandoned directories holding 150 GB. That filled the
        disk and surfaced as runs failing with "requested and 0 written",
        which reads as a capacity limit and is nothing of the sort. The
        context manager removes the directory on every exit path, the
        failures included.
        """
        with tempfile.TemporaryDirectory(prefix="seurat_arm_") as tmp:
            return self._run_in(Path(tmp), path, params, timeline, artifacts)

    def _run_in(self, work: Path, path: str, params: Any, timeline: Any,
                artifacts: Path) -> dict[str, Any]:
        artifacts.mkdir(parents=True, exist_ok=True)

        # Conversion is deliberately outside every timed step. See module docs.
        # The BPCells arms stream the h5ad straight into BPCells' directory
        # format instead, so they never hold the matrix in memory.
        counts = work / "counts"
        if not self.use_bpcells:
            _h5ad_to_rds(path, counts)

        cfg = {
            "counts_stem": str(counts),
            "h5ad_path": str(work / "bpcells_input.h5ad"),
            "artifacts_dir": str(artifacts),
            "use_bpcells": self.use_bpcells,
            "bpcells_dir": str(work / "bpcells"),
            "n_hvg": params.n_hvg,
            "n_pcs": params.n_pcs,
            "n_neighbors": params.n_neighbors,
            "resolution": params.resolution,
            "target_sum": params.target_sum,
            "min_genes": params.min_genes,
            "seed": params.seed,
            "with_umap": bool(params.with_umap),
            "threads": params.threads,
            "tuned": getattr(self, "tuned", False),
        }
        cfg_path = work / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        # BPCells needs its own on-disk directory, built in a separate Rscript
        # call so the conversion sits outside the timed region — the same
        # treatment every other arm's input gets. Without this,
        # `open_matrix_dir` fails with "not a valid file path".
        if self.use_bpcells:
            _bpcells_readable(path, work / "bpcells_input.h5ad")
            _cenv = dict(os.environ)
            _cenv.setdefault("R_MAX_VSIZE", "100Gb")
            conv = subprocess.run(
                ["Rscript", "--vanilla", str(R_BPCELLS_FROM_H5AD), str(cfg_path)],
                capture_output=True, text=True, check=False, env=_cenv,
            )
            if conv.returncode != 0:
                raise RuntimeError(
                    "BPCells conversion failed:\n"
                    + "\n".join(conv.stderr.splitlines()[-20:])
                )

        # R caps its own vector heap independently of the machine. On macOS the
        # default tripped at 200K cells with "vector memory limit of 16.0 Gb
        # reached", while the process held only 3.25 GB RSS and swapped 0 MB —
        # a configuration limit, not a memory wall. Reporting that as Seurat's
        # capability boundary would have been wrong, and any Seurat user who
        # met it would raise the limit, so the arm raises it too. The machine's
        # real memory still binds; this only stops R refusing before it.
        env = dict(os.environ)
        env.setdefault("R_MAX_VSIZE", "100Gb")

        t0 = timeline.now()
        code, stdout, stderr, peak = _run_r(self.r_script, cfg_path, env)
        _replay_steps(stderr, timeline, t0)

        if code != 0:
            raise RuntimeError(
                f"Rscript failed ({code}). Last stderr:\n"
                + "\n".join(stderr.splitlines()[-25:])
            )

        summary = json.loads(stdout.strip().splitlines()[-1])
        arr = _csv_to_npy(artifacts)
        files = write_artifacts(
            artifacts,
            embedding=arr["embedding"], labels=arr["labels"],
            hvg_names=arr["hvg_names"], obs_names=arr["obs_names"],
            umap=arr["umap"],
        )
        return {
            "n_obs_after": int(summary["n_obs_after"]),
            "n_vars_after": int(summary["n_vars_after"]),
            "n_clusters": int(summary["n_clusters"]),
            "storage": "bpcells" if self.use_bpcells else "ram",
            # The R process's own peak footprint (see _run_r); the record's
            # headline memory figure for this arm.
            "footprint_bytes": int(peak),
            "bpcells_version": summary.get("bpcells_version"),
            "seurat_version": summary.get("seurat_version"),
            "r_blas": summary.get("blas"),
            "r_version": summary.get("r_version"),
            # The heap ceiling in force and how close the run came to it.
            # `r_vsize_limit_mb` is what separates a record measured under
            # R's macOS default from one measured under the raised
            # ceiling; without it the two are indistinguishable.
            "r_max_vsize_env": summary.get("r_max_vsize_env"),
            "r_vsize_limit_mb": summary.get("r_vsize_limit_mb"),
            "r_vcells_peak_mb": summary.get("r_vcells_peak_mb"),
            # USS is Python-only; the parent sampler supplies RSS for the tree.
            "uss_available": False,
            "artifacts": files,
        }


#: The heap ceiling applies to every Seurat arm, so it is stated on every
#: Seurat record rather than on one of the three.
_R_HEAP_NOTE = (
    "R_MAX_VSIZE is raised to 100Gb for every Seurat arm. R caps its own "
    "vector heap independently of the machine, and the macOS default refused "
    "at 200K cells ('vector memory limit of 16.0 Gb reached') while the "
    "process held 3.25 GB RSS and had swapped 0 MB. Left unraised, Seurat "
    "would have been recorded as failing at a size it completes in 642 s "
    "using 4.71 GB. That would have been a configuration artefact reported "
    "as a capability limit, in our own favour. Each record carries the "
    "ceiling that was in force (r_vsize_limit_mb) and the peak heap the run "
    "reached (r_vcells_peak_mb)."
)


class SeuratDefault(_SeuratBase):
    name = "seurat-default"
    label = "Seurat (defaults)"
    tuned = False
    tuning_notes = (
        "Seurat's tutorial defaults, untouched, which differ from scanpy's on "
        "three axes and are left differing: k.param=20 (scanpy 15), "
        "resolution=0.8 (scanpy 1.0), algorithm=1 Louvain (scanpy Leiden). "
        "LogNormalize, vst variable features, ScaleData on the variable "
        "features, RunPCA, FindNeighbors, FindClusters. Only the thread count "
        "is pinned to the campaign budget, because an arm given more workers "
        "measures the launcher rather than the implementation. Expect a "
        "different cluster count from the other arms — that is the result, "
        "not a defect."
    ) + " " + _R_HEAP_NOTE


class SeuratTuned(_SeuratBase):
    name = "seurat-tuned"
    label = "Seurat (tuned)"
    tuned = True
    tuning_notes = (
        "Matched to scanpy-tuned wherever Seurat has an equivalent knob: "
        "Leiden via algorithm=4 with method='igraph' (the same backend "
        "scanpy-tuned uses), and the same n_pcs, k.param, resolution and "
        "seed. "
        "NOT matched, deliberately: Seurat clusters on a pruned *shared* "
        "nearest-neighbour graph while scanpy clusters on a fuzzy kNN graph. "
        "That is a methodological difference between the tools rather than a "
        "parameter, and forcing either to imitate the other would measure a "
        "pipeline nobody runs. It is the most likely source of any remaining "
        "cluster-count difference and is reported as such. "
        "Note also that R links OpenBLAS here while numpy links Apple "
        "Accelerate, so a Seurat-vs-scanpy timing difference on this hardware "
        "is partly a BLAS difference — recorded per run and stated in "
        "Methods."
    ) + " " + _R_HEAP_NOTE


class SeuratBPCells(_SeuratBase):
    name = "seurat-bpcells"
    label = "Seurat (BPCells, disk-backed)"
    tuned = True
    use_bpcells = True
    tuning_notes = (
        "Seurat v5 backed by BPCells, the disk-backed path its documentation "
        "recommends for data larger than memory. This is the nearest peer to "
        "Polariseq's out-of-core mode. Conversion to the BPCells directory "
        "format happens before the timed region, as with every other arm's "
        "input. The sketch-based workflow is NOT used: it subsamples, so it "
        "answers a different question than the exact pipeline every other arm "
        "runs."
    ) + " " + _R_HEAP_NOTE


class BPCellsNative(_SeuratBase):
    name = "bpcells-native"
    label = "BPCells (native, disk-backed)"
    tuned = True
    use_bpcells = True
    r_script = R_BPCELLS_NATIVE
    tuning_notes = (
        "BPCells' own pipeline without Seurat (R/bpcells_native.R): streamed "
        "QC and normalisation over the bit-packed matrix on disk, scanpy's "
        "seurat-flavour HVG from BPCells' streamed row statistics, the HVG "
        "subset materialised in memory as the BPCells tutorial does, PCA of "
        "the centred log data with BPCells::svds, kNN with knn_hnsw, a "
        "UMAP-style geodesic graph and cluster_graph_leiden at the shared "
        "resolution (the graph is handed to igraph from its edge list, because the wrapper's adjacency conversion densifies the graph and exhausted memory at 50K cells; the clustering call is unchanged). Per-gene scaling, which the BPCells tutorial adds, is "
        "omitted because the fixed pipeline omits it for every arm. Conversion "
        "from h5ad to the BPCells directory is streamed and untimed."
    )
