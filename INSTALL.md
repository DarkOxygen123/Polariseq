# Installing Polariseq

Polariseq is a Python package with a compiled Rust core. It is installed from its GitHub repository: pip
downloads the code and compiles the core during installation.

## What the build needs

| | |
|---|---|
| Python | 3.10 to 3.14, 64-bit |
| C compiler | Xcode Command Line Tools (macOS), Build Tools for Visual Studio with "Desktop development with C++" (Windows), gcc (Linux; usually present, else build-essential or your distribution's equivalent) |
| Rust | the stable toolchain. pip installs it for the build when it is missing; to have your own, [rustup.rs](https://rustup.rs) |
| CMake | 3.16 or newer, which builds the bundled HDF5. pip fetches it for the build |
| Git | to fetch the repository |

So the C compiler, Python and Git are the tools to have; pip takes care of the rest.

## 1. With pip, from GitHub (recommended)

### macOS

```bash
xcode-select --install                      # the C compiler, once (skip if already installed)
python3 -m venv polariseq-env
source polariseq-env/bin/activate
pip install "git+https://github.com/DarkOxygen123/Polariseq.git"
```

### Windows

Install, once: 64-bit Python 3.10 to 3.14 from python.org, Git for Windows, and Build Tools for Visual
Studio with "Desktop development with C++" (`winget install Microsoft.VisualStudio.2022.BuildTools`, then
choose that workload). Then, in PowerShell:

```powershell
py -m venv polariseq-env
polariseq-env\Scripts\activate
pip install "git+https://github.com/DarkOxygen123/Polariseq.git"
```

If the build says the linker `link.exe` is not found, run the same lines from a *Developer PowerShell for
VS*.

### Linux

```bash
python3 -m venv polariseq-env
source polariseq-env/bin/activate
pip install "git+https://github.com/DarkOxygen123/Polariseq.git"
```

On an older system (glibc older than 2.28: CentOS or RHEL 7, many computing clusters), add
`--prefer-binary`. The newest NumPy, pandas and SciPy publish no ready-made packages for those systems, and
pip would otherwise try to compile them; with the flag it takes the newest versions that have one:

```bash
pip install --prefer-binary "git+https://github.com/DarkOxygen123/Polariseq.git"
```

### All systems

The first installation compiles HDF5 and the Rust core and takes a few minutes. With
[uv](https://docs.astral.sh/uv/): `uv venv && uv pip install "git+https://github.com/DarkOxygen123/Polariseq.git"`.
With conda: create an environment with Python 3.10 or newer, activate it, then run the `pip install` line.

Check the install:

```bash
python -c "import polariseq as ps; print(ps.__version__); print(ps.resources())"
```

## 2. Extras: add what you need

The install above is the analysis alone: reading `.h5ad` and 10x files, planning, quality control,
normalization, gene selection, PCA, the neighbour graph, Leiden clustering (multi-level Louvain with
`method="fast"`), UMAP, Paris, PAGA, marker genes, doublets (Scrublet), integration (Harmony) and comparisons
between conditions, with NumPy and pandas as its only Python dependencies. Each extra adds one purpose:

| extra | installs | needed for |
|---|---|---|
| `plot` | Matplotlib | every figure: `ps.pl.umap`, dot plots, violins, heatmaps, volcano plots, QC and overview figures, and the figures examples 02, 04, 05 and 06 write |
| `anndata` | AnnData, SciPy | `adata.to_anndata()` and writing `.h5ad`, to continue in Scanpy, scVI or any tool built on AnnData; example 05 reads its data with it |
| `scipy` | SciPy | results as SciPy sparse matrices, to hand to other tools: the neighbour graph in `adata.obsp`, `.to_scipy()`, `adata.sample()` |
| `markers` | pyreadr, openpyxl | the marker databases fetched on request: `ps.tl.auto_annotate(adata, source="accordion")` (also `"cellmarker"`, `"sctype"`) |
| `celltypist` | CellTypist, AnnData, SciPy | cell-type annotation with CellTypist models: `ps.tl.auto_annotate(adata, source="celltypist")` |
| `all` | all of the above: Matplotlib, AnnData, SciPy, CellTypist, pyreadr, openpyxl | everything; the same as `[plot,anndata,scipy,markers,celltypist]`. The largest install, since CellTypist brings Scanpy and scikit-learn |

Name one or several in brackets:

```bash
pip install "polariseq[plot] @ git+https://github.com/DarkOxygen123/Polariseq.git"            # analysis and figures
pip install "polariseq[plot,anndata] @ git+https://github.com/DarkOxygen123/Polariseq.git"    # and the hand-over to Scanpy
pip install "polariseq[all] @ git+https://github.com/DarkOxygen123/Polariseq.git"             # everything
```

On Windows the same lines work in PowerShell. An extra can be added later the same way (pip keeps the
build it has), or by installing the package itself (`pip install matplotlib`). Without SciPy, the
neighbour graph in `adata.obsp` holds its CSR arrays (`indptr`, `indices`, `data`), gives the same clusters
and layouts, and its `to_scipy()` converts it once SciPy is installed. A function that needs a missing extra
stops with a message that names it.

## 3. On a cluster node or a machine without internet

If the node can reach GitHub and has a C compiler, section 1 is all there is (on an older system, with
`--prefer-binary`). If it cannot, build a wheel on a machine with internet and the same operating system,
processor and Python version (for Linux, best a machine of the same distribution as the node, or the node's
login host), copy it over and install it from the file:

```bash
# on a machine with internet: a wheel of Polariseq and wheels of its dependencies
pip wheel --prefer-binary "git+https://github.com/DarkOxygen123/Polariseq.git" --wheel-dir wheels

# copy the wheels/ folder to the node, then on the node:
pip install --no-index --find-links wheels polariseq
```

Add the extras you need to both lines (`"polariseq[plot] @ git+..."` and `"polariseq[plot]"`). A wheel
installs without any build tools. [examples/NODE_SETUP.md](examples/NODE_SETUP.md) covers building on the
node itself, sizing a run, schedulers (Slurm) and reading the logs.

## 4. From source: the complete build

To change the code, run the tests or build wheels for other computers. Here maturin runs the build itself,
so all four build tools must be installed: Python 3.10 to 3.14 (64-bit), a C compiler (section 1), Rust
from [rustup.rs](https://rustup.rs) (`rustc --version` to check) and CMake 3.16 or newer (`pip install
cmake`, or your system's).

macOS and Linux:

```bash
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
python3 -m venv .venv
source .venv/bin/activate
pip install maturin cmake
maturin develop --release --extras all     # builds the Rust core and installs Polariseq with every extra
pip install pytest scikit-learn             # for the tests
python -m pytest                            # from the clone's root folder
```

Windows (PowerShell, in the clone's folder; from a *Developer PowerShell for VS* if `link.exe` is not
found):

```powershell
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
py -m venv .venv
.venv\Scripts\activate
pip install maturin cmake
maturin develop --release --extras all
pip install pytest scikit-learn
python -m pytest
```

- `--release` matters: a debug build is 10 to 50 times slower. The first build compiles HDF5 from source
  and takes a few minutes; later builds, after a change to the code, are incremental.
- `maturin develop --release` alone installs the core; `--extras plot,anndata` installs those extras.
  On an older Linux system (section 1), set `export PIP_PREFER_BINARY=1` first, so that the extras come
  as ready-made packages.
- The tests read the files in `tests/data/`, so run them from the clone's root folder. Tests of an extra
  that is not installed are skipped.
- `maturin build --release` writes a wheel to `target/wheels/` instead of installing. The wheel installs
  with pip, without any build tools, on other computers of the same operating system, processor and Python
  version: `pip install "polariseq[all]" --find-links target/wheels` (or name the extras you need).
- `pip install ".[all]"` in the clone is the one-step alternative: pip builds a release build and installs
  it with the extras, fetching Rust and CMake itself as in section 1.

## 5. Check it works

```bash
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq   # for the examples
cd polariseq
python examples/01_quickstart.py      # a small bundled file, a few seconds
```

Examples 01 and 03 need the core only; 02, 04 and 06 the `plot` extra; 05 `plot` and `anndata`. The tests
are run from a clone (section 4).

## Troubleshooting

**The installation fails while building.** Check `python --version` (3.10 to 3.14), that the Python is
64-bit, and that a C compiler is installed (section 1). On an older Linux system, add `--prefer-binary`.

**pip tries to compile NumPy, pandas or SciPy.** The system is too old for their newest ready-made packages;
add `--prefer-binary` (section 1, Linux).

**`ModuleNotFoundError: No module named 'polariseq._polariseq'`.** You are running Python from inside a
source checkout that was not built. Run `maturin develop --release` there, or run your script from another
folder.

**`error: linker 'cc' not found`, or `link.exe` not found.** No C compiler; see section 1. On Windows, run
the build from a *Developer PowerShell for VS*.

**`ImportError: Polariseq's plots need Matplotlib`, or a function asks for SciPy.** Add the extra it names
(section 2).

**`Both VIRTUAL_ENV and CONDA_PREFIX are set` (maturin).** A virtual environment was activated inside a
conda environment. Use one of them: `conda deactivate` before activating `.venv`, or skip `.venv` and run
the section 4 commands in the activated conda environment.

**Everything is slow (source build).** The build was made without `--release`. Rebuild with it.
