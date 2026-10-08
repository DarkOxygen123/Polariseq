# Running Polariseq on another machine

For a node or server where `pip install polariseq` is not an option: you take
the code from git, build it in a virtual environment inside the clone, and run
the examples from there. No root is needed, and nothing goes into the system
Python. Everything below was run from a fresh clone.

1. [What the machine needs](#1-what-the-machine-needs)
2. [Get the code](#2-get-the-code)
3. [Set it up](#3-set-it-up) (Linux and macOS)
4. [Without access to PyPI or crates.io](#4-without-access-to-pypi-or-cratesio)
5. [A very large run](#5-a-very-large-run)
6. [Tracking time and memory](#6-tracking-time-and-memory)
7. [The figures](#7-the-figures)
8. [When something goes wrong](#8-when-something-goes-wrong)
9. [On Windows](#9-on-windows)

---

## 1. What the machine needs

| | |
|---|---|
| Python | 3.10 to 3.14 (`module load python` on most clusters) |
| git | to clone |
| C compiler | `gcc` or `cc` (`module load gcc`, or build-essential) |
| Rust | the stable toolchain from [rustup](https://rustup.rs), installed into your home, no root |
| Disk | about 3 GB for the build, then room for the run (section 5) |

The HDF5 reader is built from source, so no system HDF5 is needed. A first
release build took 2.5 minutes on an 8-core laptop and 1.2 minutes when the
dependencies were already compiled.

## 2. Get the code

The repository is private, so git needs a credential. Any one of these:

```bash
# the GitHub command line, once per machine
gh auth login
gh repo clone DarkOxygen123/Polariseq-main polariseq

# or HTTPS with a fine-grained token (Contents: read-only on this repository),
# pasted when git asks for the password, never put in the command
git clone https://github.com/DarkOxygen123/Polariseq-main.git polariseq

# or SSH, with a deploy key added to the repository
git clone git@github.com:DarkOxygen123/Polariseq-main.git polariseq

cd polariseq
```

`main` is the library and these examples. The `bench` branch adds the
benchmark harness (`git clone -b bench ...`); you do not need it to run an
analysis.

## 3. Set it up

```bash
bash examples/setup_node.sh                 # Rust already installed
bash examples/setup_node.sh --install-rust  # install Rust into ~/.cargo first
bash examples/setup_node.sh --python python3.12
```

The script checks Python, Rust and a C compiler; creates `.venv` inside the
clone; installs the dependencies and `maturin`; builds Polariseq in release
mode; and finishes by running example 06 on the bundled fixture, so a clean
finish means the install works. Run it again after a `git pull` to rebuild.

The same by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy scipy matplotlib pandas anndata psutil
maturin develop --release        # not without --release: a debug build is 10 to 50 times slower
python -c "import polariseq as ps; print(ps.get_build_info())"
```

Each later session starts with `source polariseq/.venv/bin/activate`.

## 4. Without access to PyPI or crates.io

A compute node often cannot reach the internet. Build once where you can,
then carry two things to the node: the Polariseq wheel and a folder of the
Python packages it needs.

**On a machine with internet that matches the node** (same operating system
family, CPU architecture and Python minor version, for example another login
node of the same cluster):

```bash
bash examples/setup_node.sh                        # builds and checks it
source .venv/bin/activate
maturin build --release -o ~/carry/wheels          # writes polariseq-*.whl
pip download -d ~/carry/wheelhouse numpy scipy matplotlib pandas anndata psutil
```

**On the node**, with the repository cloned or copied there and `~/carry`
copied across:

```bash
bash examples/setup_node.sh --wheel ~/carry/wheels/polariseq-*.whl \
                            --wheelhouse ~/carry/wheelhouse
```

That installs from those files only, with no network, and needs neither Rust
nor a compiler. It took 40 seconds with a 59 MB wheelhouse. A wheel is tied
to a Python minor version and a platform: `cp314` and `macosx_11_0_arm64` in
its file name mean Python 3.14 on Apple Silicon, so build on a machine whose
Python and platform match the node's.

## 5. A very large run

`examples/06_large_run.py` plans a run, does it with every step timed and
memory sampled, writes the figures, and exports the per-cell results. The
simplest way to use it is to edit the SETTINGS block at the top of the file
(input folder, output and project folders, memory, threads, parameters) and run
it with no arguments; the options below override those settings, which is what
a job script does.

```bash
source .venv/bin/activate
python examples/06_large_run.py /data/atlas.h5ad \
    --out runs/atlas \
    --project /scratch/$USER/polariseq-atlas \
    --ram-gb 64 --threads 16
```

| option | meaning |
|---|---|
| `--out` | where the figures, `report.json` and `cells.tsv.gz` go |
| `--project` | Polariseq's working files: the on-disk copy of the counts and the results. Use a fast local disk. Not a home directory with a quota, and not a network file system |
| `--ram-gb` | the memory Polariseq may use. Pin it; left alone it follows whatever is free, and two runs then disagree |
| `--threads` | worker threads. The physical cores are the useful number. Pin it as you pin the memory: the plan is made for this value |
| `--mode` | `auto` (default), `memory` or `out-of-core` |
| `--min-genes` | fewest genes for a cell to be kept. A property of your data: 200 suits a droplet dataset with about 30,000 genes. The script warns if it removes most cells |
| `--n-hvg` | highly variable genes (2000); PCA and every later step use only these |
| `--umap` / `--no-umap` | compute the UMAP and draw it (on by default). It is the slowest step at scale; `--no-umap` for a first pass |
| `--color-by` | input columns to colour extra UMAPs by, comma-separated |
| `--keep-obs` | columns of the input's cell table to copy into `cells.tsv.gz`, comma-separated (an id, cell types, donors), for example `--keep-obs soma_joinid,cell_type,donor_id`. A name the input lacks is declined at the start |

**Try it small first.** Run it with no file (it uses the bundled fixture), and
then on a random tenth of your data, before the real run. The plan it prints
is read from the file's description alone, so it is instant for any size:

```
plan, before anything runs:
  in memory    peak   22.16 GB, limited by 'hvg subset', 32,768 cells per block: does NOT fit (needs a budget of 22.2 GB)
  out of core  peak    5.61 GB, limited by 'leiden', 32,768 cells per block: fits
  4,062,980 cells, 63,561 genes, 2,631,157,163 stored values (the on-disk copy needs about 22 GB)
```

That is the plan of the 4.06 million cell fetal atlas at a 6 GB budget. If
nothing fits, the script stops and says so before it reads a byte. Raise
`--ram-gb` or use a larger machine.

**What a run needs.** Measured on a MacBook Air (Apple M2, 8 GB of memory),
four threads, a 6 GB budget, quality control through clustering, out of core,
under the default workflow:

| dataset | cells | stored values | time | peak resident memory | on-disk copy |
|---|---:|---:|---:|---:|---:|
| mouse brain (10x Genomics) | 1.3 million | 2.6 billion | 3.3 min | 3.1 GB | 4.6 GB |
| human fetal atlas (Cao et al. 2020) | 4.06 million | 2.63 billion | 5.3 min | 3.8 GB | 4.0 GB |

The on-disk copy is the block store in the project folder (compact layout,
about 1.5 to 1.8 bytes per stored value). `ps.plan()` prints the same figures
for your own file before anything runs.

Rules of thumb from those runs. Treat them as a guide to the order of
magnitude; the plan and a run on a subset are the authority.

- **Disk.** The copy of the counts is written compact and lossless: about one
  byte per stored value on count data (measured 0.96 on scTab and 1.24 on the
  fetal atlas with the same method; the plan reserves 2), against 8 for the
  plain layout (`POLARISEQ_STORE=plain`). PCA writes a copy of the selected
  genes too, and the results add about half a gigabyte per million cells. The
  plan prints the project folder's peak; the example checks it against the free
  disk before it starts.
- **Memory.** What grows with the number of cells is the per-cell results and
  the graph steps, about 2 GB per million cells at the sizes measured. The
  matrix itself never needs to fit. At the largest size the process held 0.5 GB
  more than the 6 GB budget it was given, so ask the scheduler for at least a
  quarter more memory than `--ram-gb`.
- **Time.** Between 1 and 4 million cells it grew a little faster than the
  number of cells (about the 1.1 power), from 53 to 64 microseconds per cell on
  the laptop. UMAP and marker genes are not in those times. More cores shorten
  it. Time a tenth of your data and scale it up with those figures in mind.
- **The first step reads the whole input.** Out of core, the first step includes
  the one-time copy of the counts to `--project`, a pass over the whole input
  file. It is reused if you run again on the same unchanged file.

**Under a scheduler** (SLURM here; the idea is the same elsewhere). Ask for
more memory than `--ram-gb`, and keep `--out` and `--project` on disks that
outlive the job:

```bash
#!/bin/bash
#SBATCH --job-name=polariseq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=polariseq-%j.log
source .venv/bin/activate
python examples/06_large_run.py /data/atlas.h5ad --out runs/atlas \
    --project /scratch/$USER/polariseq-atlas \
    --ram-gb 64 --threads "$SLURM_CPUS_PER_TASK" --umap
```

On a machine without a scheduler, `nohup python examples/06_large_run.py ...
> run.log 2>&1 &` (or `tmux`) keeps it running after you log out; `tail -f run.log`
follows it.

## 6. Tracking time and memory

`06_large_run.py` does this for you with `examples/_resources.py`, which has
no dependency beyond the standard library (it uses `psutil` if installed). It
samples twice a second: the memory the process holds, the memory the machine
still has free, and the free space on the disk Polariseq writes to. Use it
around your own code the same way:

```python
import sys; sys.path.insert(0, "examples")
from _resources import ResourceLog
import polariseq as ps

with ResourceLog(watch="/scratch/me/polariseq-atlas") as log:
    with log.step("pca"):
        ps.pp.pca(adata, n_components=50)
    with log.step("neighbors"):
        ps.pp.neighbors(adata, n_neighbors=15)

print(log.summary())      # seconds, peak memory, lowest free memory, disk written, every step
log.save("out/")          # resources.csv and resources.png
```

What a run leaves in `--out`:

| file | what it shows |
|---|---|
| `report.json` | settings, the plan, seconds and peak memory per step, total time, disk used by each part of the project, the machine and the Polariseq build |
| `resources.csv` | one row every half second: seconds, memory held, free memory, free disk, the step running |
| `resources.png` | memory held, free memory and disk written against time, each step shaded |

Reading them:

- **Peak memory held** is the process's resident size. `peak_memory_gb.held_os`
  is the operating system's own record of its highest value and
  `held_sampled` is the highest of the twice-a-second samples. The first is the
  one to trust, because a short spike between two samples is missed by the
  second (on a 300,000 cell test they were 1.39 and 1.28 GB).
- **Free memory** is what the machine could still hand out. A run that leaves
  it near zero is about to swap or be killed.
- **On macOS** the system compresses memory under pressure, so a process can
  hold more than its resident size shows (Activity Monitor's "memory"
  column is the physical footprint, which counts compressed pages). On
  Linux and Windows the resident size is the figure to watch.
- **Under a scheduler** the limit is the job's memory request, enforced by the
  kernel. If a job is killed, compare `held_os` with `--mem`.

Without any of this, the system's own tools give the peak of any command:
`/usr/bin/time -v python ...` on Linux (look for "Maximum resident set size"),
`/usr/bin/time -l python ...` on macOS, and `seff <jobid>` or `sacct -o
JobID,MaxRSS,Elapsed` after a SLURM job.

## 7. The figures

Every run of example 06 writes:

| figure | what it shows |
|---|---|
| `qc.png` | cells by genes detected and by counts: look at it to choose `--min-genes` |
| `pca.png` | variance explained by each principal component |
| `clusters.png` | cells per cluster, largest first |
| `umap.png` | the embedding coloured by cluster; with many clusters the largest are named on the data instead of a legend; `umap_<column>.png` for each `--color-by` column |
| `resources.png` | memory, free memory and disk against time (section 6) |

Every figure covers every cell. The histograms use bins spaced evenly on a log
scale, and above a million cells the UMAP is drawn as an image of binned cells
(each pixel in the colour of most of its cells, its strength from how many
there are), which takes seconds where a scatter of tens of millions of points
takes minutes and turns into a blot. The same figures come from
`ps.pl.overview(adata, out)` on any analysis. `run_summary.md` states the run
and the machine it ran on for a methods section.

`cells.tsv.gz` has one row per cell kept (name, file, genes, counts, cluster, UMAP
and any columns you named with `--keep-obs`). Cell names can repeat across
files, so a cell is identified by `dataset` and `cell` together, or by an id
column you keep. With it you can draw anything else from it in Python or R without Polariseq. For
the full set of Polariseq plots (dot plots, heat maps, violins, themes,
multi-panel figures) see `examples/04_plotting.py`; they take any object
the pipeline returns.

## 8. When something goes wrong

| symptom | cause and fix |
|---|---|
| `declined: no mode fits this budget` | the plan found no way to stay within `--ram-gb`. Raise it, free memory, or use a larger machine. It is declined before anything is read |
| `WARNING: --min-genes 200 removed 98% of the cells` | the threshold does not suit the data. Lower it (see `qc.png`) |
| `declined: X of ... does not hold raw counts` | the files hold normalized or log-transformed values. Polariseq reads X; write the raw counts to X (the message names `layers/counts` or `raw/X` if the file has them), or pass `--not-counts` to analyse it anyway |
| `declined: the datasets share no genes` | the files name their genes differently (symbols in one, Ensembl ids in another); make them match. A `WARNING: only N genes are in every file` means the same, partly |
| `Rust is not installed` | re-run `setup_node.sh` with `--install-rust`, or install from rustup.rs, or build a wheel elsewhere (section 4) |
| `linker 'cc' not found` | no C compiler: `module load gcc` or install build-essential |
| `No module named polariseq._polariseq` | the extension was not built into this environment: activate `.venv` and run `maturin develop --release` |
| everything is slow | a build without `--release` |
| the job is killed with no Python error | out of memory under the scheduler: raise `--mem` to at least a quarter above `--ram-gb`, or lower `--ram-gb` |
| `No space left on device` | `--project` is on a small disk: compare the plan's "project folder at most" with the free space there |
| a network file system is slow | put `--project` on a local or scratch disk; the counts are read back in blocks |

---

## 9. On Windows

`setup_node.sh` is for Linux and macOS. On a Windows machine, whether you sit
at it or reach it by Remote Desktop or SSH, use the PowerShell script that
ships on the `bench` branch (`bench` is `main` plus the benchmark harness;
the examples are the same). It has run on a real Windows laptop. It keeps
uv, Python, Rust and CMake inside the clone and installs only one system
component, the Visual Studio C++ build tools, which Windows asks permission
for once (an administrator prompt) and which take about 3 GB.

**Check the machine first** (PowerShell):

```powershell
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB      # RAM, GB
(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum   # cores
(Get-PSDrive D).Free / 1GB                                            # free disk on D:, GB
```

**Set up** (git must be installed: Git for Windows; sign in once so it can read
the private repository, for example `gh auth login`, or accept the browser
prompt Git Credential Manager shows on the first clone):

```powershell
git clone -b bench https://github.com/DarkOxygen123/Polariseq-main.git D:\polariseq
cd D:\polariseq
powershell -ExecutionPolicy Bypass -File benchmarks\windows\setup.ps1 -SkipTests
```

Clone to a short path on the big drive, not into OneDrive: the script refuses
OneDrive (it would try to upload the data) and warns about long paths (the
HDF5 build nests folders deeply and can hit Windows' 260-character limit). The
first build takes 10 to 20 minutes, because HDF5 is compiled; setup finishes
with a one-minute check. Nothing else needs installing: the virtual environment
is `.venv\` in the clone.

**Run** with that environment's Python:

```powershell
.\.venv\Scripts\python.exe examples\06_large_run.py D:\data\atlas.h5ad `
    --out D:\runs\atlas --project D:\polariseq-work --ram-gb 48 --threads 16 --umap
```

**Unattended.** Disconnecting from Remote Desktop leaves your session, and what
runs in it, going; signing out, or the machine going to sleep, ends it. Start
the run detached, with its output in files, and turn sleep off while it runs:

```powershell
powercfg /change standby-timeout-ac 0
Start-Process -FilePath .\.venv\Scripts\python.exe -WindowStyle Hidden `
    -ArgumentList 'examples\06_large_run.py','D:\data\atlas.h5ad','--out','D:\runs\atlas',
                  '--project','D:\polariseq-work','--ram-gb','48','--threads','16' `
    -RedirectStandardOutput D:\runs\atlas.log -RedirectStandardError D:\runs\atlas.err
Get-Content D:\runs\atlas.log -Wait        # follow it; Ctrl+C stops following, not the run
```

On Windows the memory the system will hand out is capped by installed memory
plus the page file (the commit limit). Leave the page file system-managed. The
resident size your run logs is the working set; `peak_memory_gb.held_os` in
`report.json` is Windows' own peak working set.

**Data from the CELLxGENE Census.** The Census is read with `tiledbsoma`, which
PyPI offers for Linux and macOS only (no Windows wheels as of version 2.3.0).
Query it from a Linux or macOS machine, or from WSL2 on the Windows machine,
write the cells to `.h5ad` files, and let Polariseq read those on Windows.

