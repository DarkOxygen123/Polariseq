"""Scarf: the out-of-core Python toolkit closest in aim to Polariseq.

Scarf (Dhapola et al., Nat. Commun. 2022) stores counts in chunked Zarr and
streams them through incremental PCA, an HNSW graph and Leiden. It runs in
its own interpreter (``.venv-scarf``: Scarf 0.32 pins numpy 1.26 and numba
<= 0.61, neither available for the Python 3.14 the other arms use), driven
as a subprocess like the R arms; memory is that subprocess's own peak
footprint.

Declared differences, each one Scarf's design rather than a setting:

* **Conversion is not timed.** Scarf analyses its own Zarr store, as BPCells
  analyses its directory; building it from the h5ad is setup, in a separate
  process. (Polariseq's out-of-core timings *include* writing its block
  store.)
* **Input reading is patched.** Scarf 0.32.3's h5ad reader drops the last
  cell (an off-by-one in ``consume_group``) and assumes CSR; the conversion
  uses a corrected reader (``scarf_pipeline._consume_rows``). This touches
  only the untimed conversion.
* **Highly variable genes** are Scarf's own (lowess-detrended variance),
  the only method it offers, with its default gene-name blacklist disabled
  because no other arm has one.
* **Warm-up.** Scarf compiles Numba code on first use; one untimed pass of
  the pipeline over a 12,000-cell synthetic store precedes the timed run, as
  for Scanpy.
* **PCA and the neighbour graph are one call** (``make_graph``); the time
  is reported under ``pca`` and ``neighbors`` is near zero.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .seurat_arm import _bpcells_readable, _replay_steps, run_tracked

REPO = Path(__file__).resolve().parents[3]
SCARF_PY = REPO / ".venv-scarf" / "bin" / "python"
SCRIPT = Path(__file__).parent / "scarf_pipeline.py"


def _synthetic_h5ad(path: Path) -> None:
    """12,000 cells x 3,000 genes of sparse integer counts (warm-up input)."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.random(12_000, 3_000, density=0.08, format="csr", random_state=rng,
                  dtype=np.float32)
    x.data = np.ceil(x.data * 10.0).astype(np.float32)
    ad.AnnData(x, obs=pd.DataFrame(index=[f"c{i}" for i in range(x.shape[0])]),
               var=pd.DataFrame(index=[f"g{i}" for i in range(x.shape[1])])).write_h5ad(path)


class ScarfArm:
    name = "scarf"
    label = "Scarf (Zarr, out-of-core)"
    tuning_notes = __doc__.split("Declared differences, each one Scarf's design rather than a setting:")[1].strip()

    def run(self, path: str, params: Any, timeline: Any, artifacts: Path) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="scarf_arm_", dir=os.environ.get("TMPDIR")) as tmp:
            return self._run_in(Path(tmp), path, params, timeline, artifacts)

    def _run_in(self, work: Path, path: str, params: Any, timeline: Any,
                artifacts: Path) -> dict[str, Any]:
        artifacts.mkdir(parents=True, exist_ok=True)
        # Scarf's reader, like BPCells', expects string indices, not
        # anndata 0.13's nullable-string groups: same no-copy shim.
        link = work / "input.h5ad"
        _bpcells_readable(path, link)
        cfg = {
            "h5ad_path": str(link), "zarr_dir": str(work / "data.zarr"),
            "artifacts_dir": str(artifacts), "threads": params.threads,
            "min_genes": params.min_genes, "n_hvg": params.n_hvg,
            "n_pcs": params.n_pcs, "n_neighbors": params.n_neighbors,
            "resolution": params.resolution, "seed": params.seed,
            "with_umap": bool(getattr(params, "with_umap", False)),
        }
        cfg_path = work / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(params.threads)

        code, _, err, _ = run_tracked([str(SCARF_PY), str(SCRIPT), "convert", str(cfg_path)], env)
        if code != 0:
            raise RuntimeError("Scarf conversion failed:\n" + "\n".join(err.splitlines()[-20:]))

        # A small synthetic store for the untimed warm-up (see scarf_pipeline).
        warm_cfg = dict(cfg, h5ad_path=str(work / "warm_link.h5ad"),
                        zarr_dir=str(work / "warm.zarr"))
        _synthetic_h5ad(work / "warm.h5ad")
        _bpcells_readable(str(work / "warm.h5ad"), work / "warm_link.h5ad")
        warm_path = work / "warm_config.json"
        warm_path.write_text(json.dumps(warm_cfg))
        code, _, err, _ = run_tracked([str(SCARF_PY), str(SCRIPT), "convert", str(warm_path)], env)
        if code != 0:
            raise RuntimeError("Scarf warm-up conversion failed:\n" + "\n".join(err.splitlines()[-20:]))
        cfg["warm_zarr"] = warm_cfg["zarr_dir"]
        cfg_path.write_text(json.dumps(cfg))

        t0 = timeline.now()
        code, out, err, peak = run_tracked([str(SCARF_PY), str(SCRIPT), "run", str(cfg_path)], env)
        _replay_steps(err, timeline, t0)
        if code != 0:
            raise RuntimeError(f"Scarf failed ({code}):\n" + "\n".join(err.splitlines()[-25:]))
        summary = json.loads(out.strip().splitlines()[-1])
        files = {k: str(artifacts / f) for k, f in
                 (("labels", "labels.npy"), ("hvg", "hvg.txt"), ("obs_names", "obs_names.txt"))}
        return {
            "n_obs_after": int(summary["n_obs_after"]),
            "n_vars_after": int(summary["n_vars_after"]),
            "n_clusters": int(summary["n_clusters"]),
            "storage": "zarr",
            "scarf_version": summary.get("scarf_version"),
            "footprint_bytes": int(peak),
            "uss_available": False,
            "artifacts": files,
        }
