"""Data for the examples: open your own file, or download a small public one.

``load(path)`` opens a ``.h5ad`` file, a 10x Genomics ``.h5`` file or a 10x
matrix folder (matrix.mtx, barcodes and genes/features). With an empty path it
downloads the 10x PBMC 3k dataset (about 8 MB, once) into ``examples/data/``.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import polariseq as ps

DATA = Path(__file__).resolve().parent / "data"
PBMC3K_URL = ("https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
              "pbmc3k_filtered_gene_bc_matrices.tar.gz")


def fetch_pbmc3k() -> Path:
    """The 10x Genomics PBMC 3k matrix folder, downloaded on first use."""
    folder = DATA / "pbmc3k"
    found = list(folder.rglob("matrix.mtx*")) if folder.exists() else []
    if found:
        return found[0].parent
    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / "pbmc3k.tar.gz"
    if not tgz.exists():
        print("downloading PBMC 3k from 10x Genomics (about 8 MB) ...", flush=True)
        # the CDN refuses Python's default user agent
        req = urllib.request.Request(PBMC3K_URL, headers={"User-Agent": "polariseq-examples"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tgz, "wb") as fh:
            fh.write(r.read())
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(folder, filter="data")
        except TypeError:  # Python before 3.12
            tf.extractall(folder)
    return next(folder.rglob("matrix.mtx*")).parent


def load(path: str = "", **kw):
    """Open ``path`` with the reader its type needs; an empty path gives PBMC 3k."""
    if not path:
        return ps.read_10x_mtx(str(fetch_pbmc3k()), **kw)
    p = Path(path).expanduser()
    if p.is_dir():
        return ps.read_10x_mtx(str(p), **kw)
    if p.suffix == ".h5":
        return ps.read_10x_h5(str(p), **kw)
    return ps.read_h5ad(str(p), **kw)
