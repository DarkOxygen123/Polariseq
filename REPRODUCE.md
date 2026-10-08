# Reproducing the Polariseq benchmarks

This branch holds the Polariseq library (the same code as `main`) plus two
folders:

| folder | what |
|---|---|
| `benchmarks/` | every script that produced a measurement: data preparation, the benchmark harness, the campaigns and the follow-up analyses |
| `results/` | the measured tables, small enough for git (about 2.4 MB). [results/README.md](results/README.md) describes each file and the script that wrote it |

The input datasets are public but large, so they are not stored here. Section 2
says how to download and prepare each one. Every script writes its raw output
under `benchmarks/results/`, which git ignores. The tables in `results/` were
exported from those raw outputs.

Commands are run from the repository root unless stated otherwise.

## 1. Set up

```bash
git clone -b reproduce https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r benchmarks/requirements.txt     # the pinned tool versions used for the runs
pip install polariseq==0.1.0                   # or, to build this checkout: maturin develop --release
```

The pins in `benchmarks/requirements.txt` are the versions the runs used
(Python 3.14). Each run also records the versions it saw, so a record can be
checked against them.

Check the harness on the bundled 10,000-cell file (a few seconds):

```bash
python benchmarks/run_telemetry.py --datasets fixture --quick
```

### Optional: the other tools

The harness compares Polariseq with other tools ("arms"). Scanpy is installed
by the step above. The others are picked up only when present:

| arms | what they need |
|---|---|
| `seurat-default`, `seurat-tuned`, `seurat-bpcells`, `bpcells-native` | R with `Rscript` on the PATH, then `bash benchmarks/install_r_packages.sh` (Seurat, igraph, and BPCells from GitHub). It compiles from source and takes a while |
| `scarf` | a second environment at `.venv-scarf/` in the repository root, with Python 3.12, `scarf==0.32.3` and `numpy==1.26.4` (Scarf does not install on Python 3.14). Also used by `validate_paris.py reference` and `scarf`, which need `scikit-network==0.33.2` there |

The Kang comparison has two reference steps of its own: edgeR and limma in R
(`Rscript benchmarks/r3_kang_limma.R`, Bioconductor packages edgeR 4.10 and
limma 3.68) and the original Scrublet (`scrublet==0.2.3`, in any separate
environment).

## 2. Get the data

Everything lands under `benchmarks/results/`, where the scripts look for it.

### Four public 10x datasets (2,700 to 20,000 cells)

PBMC 3k, PBMC 10k, mouse neurons 10k and PBMC 20k, downloaded from 10x Genomics
and converted to `.h5ad` with Scanpy's own readers (about 170 MB in all):

```bash
cd benchmarks
python -c "from telemetry.datasets import fetch_all; fetch_all(('pbmc3k', 'pbmc10k', 'neuron10k', 'pbmc20k'), 'results/validation_data')"
cd ..
```

### 10x 1.3 million mouse brain cells, and its subsets

Download the file (4.2 GB) into `benchmarks/results/validation_data/`:

```bash
mkdir -p benchmarks/results/validation_data
curl -L -A "polariseq-benchmark" -o benchmarks/results/validation_data/1M_neurons_filtered_gene_bc_matrices_h5.h5 \
    https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5
```

(10x's server refuses requests without a user agent, hence `-A`.) Then build
the subsets the size ladder uses. Cells are drawn with seed 0, so the same file
and seed give the same cells everywhere:

```bash
python benchmarks/make_subsets.py --sizes 50000 100000 200000 300000 400000 600000 800000 1000000 1306127
```

This writes `brain50k.h5ad` ... `brain1306k.h5ad` (the last holds every cell,
in the seeded order) and `subsets_manifest.json`, which records the parent's
checksum and the seed. The number of stored values of each subset identifies
its cells. Built from the same file with seed 0 they are:

| subset | stored values | size on disk |
|---|---:|---:|
| brain50k | 100,488,310 | 0.8 GB |
| brain100k | 201,050,280 | 1.6 GB |
| brain200k | 401,853,182 | 3.2 GB |
| brain400k | 803,920,843 | 6.5 GB |
| brain800k | 1,607,496,131 | 12.9 GB |

The whole ladder needs about 85 GB of free disk.

### Human fetal atlas (4.06 million cells)

Cao et al., *Science* 370, eaba7721 (2020), GEO
[GSE156793](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793).
Download three files into `benchmarks/results/fetal_data/` and unpack the loom:

```bash
mkdir -p benchmarks/results/fetal_data && cd benchmarks/results/fetal_data
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156793/suppl
curl -LO $G/GSE156793_S3_gene_count.loom.gz          # unpacks to 21.4 GB
curl -LO $G/GSE156793_S1_metadata_cells.txt.gz      # the authors' annotation of every cell
curl -LO $G/GSE156793_S8_DE_gene_cells.csv.gz       # each cell type's marker genes (14 MB)
gunzip GSE156793_S3_gene_count.loom.gz
cd ../../..
python benchmarks/fetal_prepare.py all
```

`fetal_prepare.py` writes `fetal4062k.h5ad` (every cell), the nested subsets
`fetal1000k.h5ad` and `fetal2000k.h5ad` (seed 0), and `fetal_cells.csv.gz`
(each cell's published type, organ and UMAP position, which the cell-type
comparisons read). The loom can be deleted afterwards.

### Kang et al. 2018 (control and interferon-beta PBMCs)

GEO GSE96583, batch 2. Example 05 downloads it (about 60 MB) and writes the two
conditions as `.h5ad`:

```bash
python examples/05_comparing_conditions.py
```

### scTab (22.2 million cells, 100 files)

Fischer et al., *Nature Communications* (2024). There is no single
file to download: the dataset is cut from CELLxGENE Census release 2023-05-15
by the authors' own notebook,
[`notebooks/store_creation/01_download_data.ipynb`](https://github.com/theislab/scTab/tree/devel/notebooks/store_creation)
in [theislab/scTab](https://github.com/theislab/scTab) (branch `devel`). Run
it with two edits: its output folder set to `benchmarks/results/scTab/h5ad`, and
100 chunks. It needs `cellxgene-census==1.16.2` and `tiledbsoma==1.15.7`,
which install on Linux and macOS only (on Windows, use WSL2 and copy the files
over). The result is 100 `.h5ad` files, 22,190,622 cells, 19,331 genes and
46.2 billion stored values (about 350 GB).

## 3. Run the benchmarks

Two machines were used, and each table in `results/` says which:

* **Laptop**: MacBook Air M2, 8 GB of memory, macOS. 4 threads, a 6 GB budget
  for Polariseq, a 60-minute limit per run.
* **Compute node**: 2 x Intel Xeon Platinum 8268 (48 cores, 192 GB), Linux,
  Slurm. 4 threads and a 100 GB budget (8 threads for the 22.2M-cell run).
  `results/node/node.json` lists its details.

Every run is a separate process, measured by the same harness
(`benchmarks/run_telemetry.py`): the time of each step and the peak memory of
the process. On macOS that is the process's physical footprint, which also
counts memory the system has compressed. On Linux it is the resident set. Runs write one JSON record each, never overwrite one, and a
campaign that is started again skips the runs already on record.

### On a laptop: every tool over the size ladder

```bash
python benchmarks/r3_campaign.py --list      # the plan: datasets, tools, repeats
python benchmarks/r3_campaign.py             # every tool, Polariseq under Scanpy's steps
python benchmarks/r4_campaign.py             # Polariseq with its own default workflow
python benchmarks/r3_campaign.py --umap      # the other tools again, with UMAP
python benchmarks/r4_campaign.py --umap      # Polariseq with UMAP
```

Records go to `benchmarks/results/telemetry_r3`, `telemetry_r4` and
`telemetry_r4_umap`. Keep the laptop on its charger and otherwise idle while
they run. A tool that fails at a size (out of memory or over the time limit)
is not tried at the larger sizes. To write a record folder as one table:

```bash
python benchmarks/export_tables.py benchmarks/results/telemetry_r4 runs.csv --campaign "my laptop"
```

`final_campaign.py` is the earlier laptop campaign: the public datasets, the
ladder, the fetal atlas and the atlas analyses in one command.

One run of one tool on one dataset:

```bash
python benchmarks/run_telemetry.py --datasets brain200k --arms polariseq-spill,scanpy-tuned \
    --reps 3 --threads 4 --ram-gb 6
```

`polariseq-ram` is Polariseq in memory and `polariseq-spill` out of core.
`python benchmarks/run_telemetry.py --help` lists the other options.

### On a compute node: one run per job

Edit the settings block at the top of `benchmarks/slurm/node_run.sbatch` (this
repository's path, the environment, the dataset, the tool), then:

```bash
sbatch benchmarks/slurm/node_run.sbatch
```

Run each job alone on its node. Two of your own runs on one node compete for
memory bandwidth and disk. The step-resolved Polariseq runs in
`results/node/step_resolved/` used example 06 instead of the harness, with
`--mode out-of-core --threads 4 --ram-gb 100 --no-export` on `brain1306k.h5ad`
and `fetal4062k.h5ad`.

### The 22.2 million cell atlas

Edit the settings block of `benchmarks/slurm/large_run.sbatch` (input folder,
output folder, a project folder on fast storage with about 750 GB free), then:

```bash
sbatch benchmarks/slurm/large_run.sbatch
```

It runs `examples/06_large_run.py` out of core with 8 threads and a 100 GB
budget, UMAP included, and writes `run_summary.md`, `report.json`,
`resources.csv`, the figures and `cells.tsv.gz` (one row per cell, with the
published cell type). Score the clusters against the published types with:

```bash
python benchmarks/r4_score_22m.py OUT_FOLDER --out score_22m.json
```

### Cell types, embeddings and the other analyses

| question | command |
|---|---|
| Do the fetal atlas clusters recover the published cell types and markers? | `python benchmarks/fetal_biology.py polariseq --size 4062k`, then `python benchmarks/fetal_biology.py compare --size 4062k` |
| The same, analyzed organ by organ as the authors did, and the authors' balanced-sample embedding | `python benchmarks/fetal_matched.py --help` (stages `embed`, `split`, `organs`, `compare`) |
| Seed-to-seed spread, UMAP against umap-learn, subsampling, Scanpy per organ, memory budgets | `python benchmarks/r3_analyses.py agreement`, `umap`, `organ_pcs`, `subsample`, `scanpy_organs`, `budget` |
| Each tool's clusters and layout of the fetal atlas, measured one way | `python benchmarks/r4_tools_fetal.py --help` (reads node runs, or files packed by `benchmarks/figures/pack_embeddings.py`) |
| Gene selection, normalization and UMAP settings | `python benchmarks/r4_sweep.py --help`, `python benchmarks/r3_scarf_ablation.py --help` |
| How close each UMAP start comes to the graph's eigenvectors | `python benchmarks/r3_umap_cause_4m.py FETAL_H5AD FETAL_CELLS_CSV OUT` (saves the graph), then `python benchmarks/r4_convergence.py --graph OUT --out CONV` |
| Hierarchical clustering (Paris) against scikit-network and Scarf | `python benchmarks/validate_paris.py export`, then `reference` and `scarf` in `.venv-scarf`, then `compare` and `fetal` |
| Handing Polariseq's output to Scanpy (PAGA on 4.06M cells) | `python benchmarks/r3_downstream.py` |
| Scanpy's PBMC 3k tutorial, reproduced in Scanpy and repeated in Polariseq | `python benchmarks/pbmc3k_tutorial.py` |
| Comparing conditions on Kang et al. against edgeR, limma, Scrublet and the authors' genotype calls | `python benchmarks/r3_kang.py`, `Rscript benchmarks/r3_kang_limma.R`, `python benchmarks/r3_kang_agreement.py`, and `r3_kang.py --reference` in the Scrublet environment |
| Brain atlas biology at 800K and 1.3M cells against Scanpy | `python benchmarks/atlas_analysis.py --help` |

Each script's opening docstring says what it measures, its inputs and where it
writes. Scripts that hold a whole atlas in memory are meant for the compute
node or a machine with at least 64 GB of memory.

## 4. Tests

```bash
python -m pytest benchmarks/tests
```

checks the harness (records, memory sampling, the arms) and the fetal-atlas
comparison on small synthetic files. Four of the fetal tests read the authors'
marker table, so download `GSE156793_S8_DE_gene_cells.csv.gz` (section 2)
first.

## Citation and licence

Polariseq is released under GPL-3.0-or-later (see `LICENSE`). The datasets
belong to their authors: cite 10x Genomics, Cao et al. 2020, Kang et al. 2018
and Fischer et al. 2024 when you use them.
