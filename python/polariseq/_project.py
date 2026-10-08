"""The project folder: where Polariseq keeps everything it writes to disk.

    import polariseq as ps
    proj = ps.project("~/analyses/liver")    # declare it once, first
    ad = ps.read_h5ad("donor1.h5ad")         # its files go to runs/donor1/

Every file Polariseq writes during the session goes into this folder: the
on-disk copy of a dataset's counts that an out-of-core analysis reads in
blocks, the gene-subset copy PCA reads, the scratch files of a running
step, and (as the store is built) the stored results and intermediates. Nothing
is left in hidden temporary folders, and a README in the folder says what
each part is and what may be deleted.

Layout:

    <project>/
      README.md       what each part is and what may be deleted
      project.json    Polariseq version and creation date
      runs/<name>/    one folder per dataset read (named after its file)
        matrix/       on-disk copy of the counts (large; made again from
                      the original file if deleted)
        latents/      intermediate arrays, e.g. the gene-subset copy for PCA
        results/      per-cell results of steps (e.g. doublet scores), written
                      by the Rust core and read as memory-mapped views
      scratch/        temporary files of a running step
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

_CURRENT: Project | None = None

README = """# Polariseq project

Created by Polariseq {version} on {date}. Everything Polariseq writes to disk
for the analyses of this project is kept in this folder.

| Path | What it is | Can I delete it? |
|---|---|---|
| `project.json` | the project's settings | No |
| `runs/<name>/` | one folder per dataset analyzed, named after its file | See below |
| `runs/<name>/matrix/` | on-disk copy of the counts, read in blocks by an out-of-core analysis (large) | Yes, when no analysis is running; it is made again from the original file |
| `runs/<name>/latents/` | intermediate arrays (e.g. the gene-subset copy PCA reads) | Yes, when no analysis is running; the steps that made them run again |
| `runs/<name>/results/` | per-cell results of steps (e.g. doublet scores), read as memory-mapped views | Yes, when no analysis is using them; the steps that made them run again |
| `scratch/` | temporary files of a running step | Yes, when no Polariseq process is running |

Your original data files are never written or moved.

`ps.project("<this folder>").status()` lists what each part occupies, and
`.clean("scratch")`, `.clean("latents")`, `.clean("results")` or `.clean("matrix")` deletes that
part for every dataset.
"""


class ProjectNotSet(RuntimeError):
    """Polariseq has to write to disk and no project folder is declared."""


class Project:
    """A project folder. Create or open one with :func:`project`."""

    def __init__(self, path: str | Path) -> None:
        from polariseq import __version__

        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "runs").mkdir(exist_ok=True)
        self.scratch.mkdir(exist_ok=True)
        meta = self.path / "project.json"
        if not meta.exists():
            today = datetime.date.today().isoformat()
            meta.write_text(json.dumps({"polariseq": __version__, "created": today}, indent=1))
            (self.path / "README.md").write_text(README.format(version=__version__, date=today))

    @property
    def scratch(self) -> Path:
        return self.path / "scratch"

    def run_dir(self, name: str) -> Path:
        """The folder of the dataset analyzed under ``name``."""
        return self.path / "runs" / name

    def status(self) -> dict[str, int]:
        """Bytes used by each part of the project, per dataset."""
        def size(p: Path) -> int:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

        out = {"scratch": size(self.scratch)}
        for run in sorted((self.path / "runs").glob("*")):
            for part in ("matrix", "latents", "results"):
                out[f"runs/{run.name}/{part}"] = size(run / part)
        return out

    def clean(self, part: str = "scratch") -> int:
        """Delete ``scratch``, or the ``latents``, ``results`` or ``matrix``
        of every dataset. Returns the bytes freed. Run it only when no analysis
        of this project is running."""
        if part not in ("scratch", "latents", "results", "matrix"):
            raise ValueError("part must be 'scratch', 'latents', 'results' or 'matrix'")
        targets = ([self.scratch] if part == "scratch"
                   else [r / part for r in (self.path / "runs").glob("*")])
        freed = 0
        for t in targets:
            if t.exists():
                freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
                shutil.rmtree(t)
        self.scratch.mkdir(exist_ok=True)
        return freed

    def __repr__(self) -> str:
        runs = sorted(p.name for p in (self.path / "runs").glob("*"))
        return f"Project({str(self.path)!r}, runs={runs})"


def project(path: str | Path) -> Project:
    """Declare the project folder of this session, creating it if needed.

    Everything Polariseq writes to disk from now on goes into it: the
    on-disk copies an out-of-core analysis reads, the intermediates, and the
    scratch files of running steps (the spill folder of
    ``ps.process_diskbacked`` included).
    """
    global _CURRENT
    from polariseq import set_budget

    _CURRENT = Project(path)
    set_budget(spill_dir=str(_CURRENT.scratch))
    return _CURRENT


def current() -> Project | None:
    """The project declared in this session, or ``None``."""
    return _CURRENT


def require(what: str) -> Project:
    """The current project, or :class:`ProjectNotSet` naming ``what`` needed it."""
    if _CURRENT is None:
        raise ProjectNotSet(
            f"{what} writes to disk, and Polariseq keeps everything it writes in a "
            "project folder. Declare one first: ps.project('<folder>')"
        )
    return _CURRENT
