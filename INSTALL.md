# Installing Polariseq

Polariseq is a Python package with a compiled Rust core. Prebuilt packages
(wheels) are published on PyPI for the common platforms, so most users need
one command.

## 1. With pip (recommended)

```bash
python -m venv polariseq-env
source polariseq-env/bin/activate        # Windows: polariseq-env\Scripts\activate
pip install polariseq
```

Wheels exist for Python 3.10 to 3.14 on macOS (Apple Silicon and Intel),
Linux x86_64 (glibc 2.28 or newer, as in most distributions since 2018) and
Windows x86_64. On any other platform pip builds Polariseq from
source (see section 3).

With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install polariseq`.
With conda: create an environment with Python 3.10 or newer, activate it, then
`pip install polariseq`.

Check the install:

```bash
python -c "import polariseq as ps; print(ps.__version__); print(ps.resources())"
```

Optional extras:

```bash
pip install "polariseq[annotation]"    # anndata, for adata.to_anndata() and .h5ad export
pip install "polariseq[celltypist]"    # CellTypist cell-type annotation
pip install "polariseq[interactive]"   # interactive plotly figures
```

## 2. On a cluster node or a machine without internet

If the node can reach PyPI, section 1 is all there is. If it cannot, download
the wheel for the node's platform and Python version on a machine that can,
copy it over, and install it from the file:

```bash
# on a machine with internet: the wheel and its dependencies, for Linux x86_64, Python 3.12
pip download polariseq --dest wheels --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --python-version 3.12

# copy the wheels/ folder to the node, then on the node:
pip install --no-index --find-links wheels polariseq
```

[examples/NODE_SETUP.md](examples/NODE_SETUP.md) covers sizing a run,
schedulers (Slurm) and reading the logs.

## 3. From source

Needed only for a platform without a wheel, or to change the code.

| | |
|---|---|
| Python | 3.10 to 3.14 |
| Rust | stable toolchain, from [rustup.rs](https://rustup.rs) |
| C compiler | Xcode Command Line Tools (macOS), build-essential (Linux), Build Tools for Visual Studio with "Desktop development with C++" (Windows) |
| CMake | 3.16 or newer (builds the bundled HDF5) |

```bash
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release        # builds and installs into the environment
python -c "import polariseq as ps; print(ps.__version__)"
```

`--release` matters: a debug build is 10 to 50 times slower. The first build
compiles HDF5 from source and takes a few minutes; later builds are
incremental. `maturin build --release` writes a wheel to `target/wheels/`
instead. `pip install polariseq --no-binary polariseq` builds from the
source package on PyPI with the same prerequisites.

## 4. Check it works

```bash
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq   # for the examples
cd polariseq
python examples/01_quickstart.py      # a small bundled file, a few seconds
```

From a source checkout, the tests run with `pip install pytest && python -m pytest`.

## Troubleshooting

**`pip` tries to build from source and fails.** Your Python or platform has
no wheel. Check `python --version` (3.10 to 3.14) and that the Python is
64-bit, or install the prerequisites in section 3.

**`ModuleNotFoundError: No module named 'polariseq._polariseq'`.** You are
running Python from inside a source checkout that was not built. Run
`maturin develop --release` there, or run your script from another folder.

**`error: linker 'cc' not found` (source build).** No C compiler; see the
table in section 3. On Windows, run the build from a *Developer Command
Prompt* if the linker is not found.

**Everything is slow (source build).** The build was made without
`--release`. Rebuild with it.
