"""Real public datasets for independent validation.

Synthetic data cannot validate a quality claim. Measured: on generated
blobs every arm reaches ARI 1.0, because well-separated Gaussian-ish types
are trivial to cluster — so a perfect score there says nothing about whether
two implementations agree on real tissue. It also cannot validate the
*shape* of real data: dropout, the long tail of rarely-detected genes, and
the ambient-RNA background are what the HVG and neighbour steps actually
have to cope with.

So validation downloads real, public, permanently-hosted scRNA-seq. Every
dataset here is a 10x Genomics public sample: no login, no licence click,
stable URLs, and the same files the field already benchmarks on. Nothing is
committed to the repo — files land in a cache directory and are reused.

Synthetic generation remains available for the offline case, and the report
says which was used, because a reader must not have to guess.
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 10x's CDN. Stable for years; these are the sample files linked from
#: 10xgenomics.com/datasets.
_10X = "https://cf.10xgenomics.com/samples/cell-exp"

#: Some CDNs reject the stdlib default User-Agent with 403.
_USER_AGENT = "polariseq-validate/1.0 (+https://github.com/polariseq)"


@dataclass(frozen=True)
class PublicDataset:
    """One downloadable dataset, and what it is."""

    name: str
    url: str
    kind: str  # "10x_h5" | "10x_mtx_targz"
    approx_cells: int
    approx_mb: int
    description: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


#: The ladder. Sizes span a decade, which is where behaviour changes: below
#: ~10k cells fixed costs dominate (our exact PCA has a floor set by
#: n_hvg^2, not by cell count), and above it the graph stages start to.
PUBLIC_DATASETS: dict[str, PublicDataset] = {
    "pbmc3k": PublicDataset(
        "pbmc3k",
        f"{_10X}/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "10x_mtx_targz", 2_700, 8,
        "10x PBMC 3k — the field's canonical smoke-test dataset",
    ),
    "pbmc10k": PublicDataset(
        "pbmc10k",
        f"{_10X}/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
        "10x_h5", 11_000, 37,
        "10x PBMC 10k v3 — healthy donor peripheral blood",
    ),
    "neuron10k": PublicDataset(
        "neuron10k",
        f"{_10X}/3.0.0/neuron_10k_v3/neuron_10k_v3_filtered_feature_bc_matrix.h5",
        "10x_h5", 11_500, 45,
        "10x mouse neurons 10k v3 — a second tissue, so a result is not "
        "an artefact of one cell type",
    ),
    "pbmc20k": PublicDataset(
        "pbmc20k",
        f"{_10X}/6.1.0/20k_PBMC_3p_HT_nextgem_Chromium_X/"
        f"20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5",
        "10x_h5", 20_000, 79,
        "10x PBMC 20k HT — larger, newer chemistry",
    ),
    "neuron20k": PublicDataset(
        "neuron20k",
        f"{_10X}/1.3.0/1M_neurons/1M_neurons_neuron20k.h5",
        "10x_h5", 20_000, 62,
        "20k-cell subset of 10x's 1.3M mouse brain dataset",
    ),
}

#: What `validate.py` uses when nothing is specified: four real datasets
#: spanning 2.7k to 20k cells across two tissues.
DEFAULT_LADDER = ("pbmc3k", "pbmc10k", "neuron10k", "pbmc20k")


def _download(url: str, dest: Path, quiet: bool = False) -> None:
    """Stream a URL to disk, atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"    downloading {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    # An explicit User-Agent is required, not cosmetic: the CDN answers
    # Python's default `Python-urllib/3.x` with 403 while serving the same
    # URL to curl.
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dest)
    if not quiet:
        print(f" {dest.stat().st_size / 1e6:.0f} MB")


#: Datasets derived locally from a downloaded parent rather than fetched.
#: They cannot be downloaded, so a missing file is an instruction, not an
#: error to retry. Built by ``benchmarks/make_subsets.py``, which records the
#: parent, its checksum and the sampling seed into the file's ``uns``.
DERIVED_DATASETS: dict[str, str] = {
    "brain5k": "5K-cell subset of the 10x 1.3M mouse brain",
    "brain50k": "50K-cell subset of the 10x 1.3M mouse brain",
    "brain100k": "100K-cell subset of the 10x 1.3M mouse brain",
    "brain200k": "200K-cell subset of the 10x 1.3M mouse brain",
    "brain400k": "400K-cell subset of the 10x 1.3M mouse brain",
    "brain800k": "800K-cell subset of the 10x 1.3M mouse brain",
}


def fetch(name: str, cache_dir: str | Path, *, quiet: bool = False) -> Path:
    """Download and convert one dataset to `.h5ad`, returning its path.

    Cached: an existing `.h5ad` is reused, so a reviewer re-running the
    validation does not re-download hundreds of megabytes. Conversion goes
    through scanpy's own readers, so the matrix is exactly what a scanpy
    user would get — this must not be a Polariseq-specific import path.

    Locally derived subsets (see :data:`DERIVED_DATASETS`) resolve to their
    cached file if it exists. They are never downloaded, so a missing one
    raises with the command that builds it.
    """
    if name in DERIVED_DATASETS:
        h5ad = Path(cache_dir) / f"{name}.h5ad"
        if h5ad.exists():
            return h5ad
        n = "".join(c for c in name if c.isdigit())
        raise FileNotFoundError(
            f"{name} ({DERIVED_DATASETS[name]}) is not built.\n"
            f"  build it:  python benchmarks/make_subsets.py --sizes {n}000"
        )
    if name not in PUBLIC_DATASETS:
        known = sorted(set(PUBLIC_DATASETS) | set(DERIVED_DATASETS))
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    ds = PUBLIC_DATASETS[name]
    cache = Path(cache_dir)
    h5ad = cache / f"{name}.h5ad"
    if h5ad.exists():
        return h5ad

    raw = cache / ds.filename
    if not raw.exists():
        _download(ds.url, raw, quiet=quiet)

    import scanpy as sc

    if ds.kind == "10x_h5":
        adata = sc.read_10x_h5(str(raw))
    elif ds.kind == "10x_mtx_targz":
        extracted = cache / f"{name}_mtx"
        if not extracted.exists():
            with tarfile.open(raw) as tf:
                # `filter="data"` refuses absolute paths and traversal; these
                # are trusted archives but the flag costs nothing.
                try:
                    tf.extractall(extracted, filter="data")
                except TypeError:  # Python < 3.12
                    tf.extractall(extracted)  # noqa: S202
        mtx = next(p for p in extracted.rglob("matrix.mtx*")).parent
        adata = sc.read_10x_mtx(str(mtx))
    else:  # pragma: no cover - registry is closed
        raise ValueError(f"unknown kind {ds.kind!r}")

    adata.var_names_make_unique()
    adata.write_h5ad(h5ad)
    return h5ad


def fetch_all(names: tuple[str, ...], cache_dir: str | Path, *,
              quiet: bool = False) -> list[tuple[str, str]]:
    """Fetch a ladder, returning `[(label, path)]`. Failures are reported,
    not fatal — a reviewer behind a proxy should still get the rest."""
    out: list[tuple[str, str]] = []
    for name in names:
        try:
            out.append((name, str(fetch(name, cache_dir, quiet=quiet))))
        except Exception as exc:  # noqa: BLE001 - one bad mirror must not sink the run
            print(f"    [skip] {name}: {exc}")
    return out


def describe() -> dict[str, Any]:
    """The registry, for the report's provenance section."""
    return {
        n: {"url": d.url, "cells": d.approx_cells, "mb": d.approx_mb,
            "description": d.description}
        for n, d in PUBLIC_DATASETS.items()
    }
