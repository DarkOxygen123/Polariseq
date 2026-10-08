# Examples

Six scripts, in order. Each one starts with a settings block (input file,
output folder, memory, threads): set your paths there, save, and run it from
any folder.

| script | what it shows | data |
|---|---|---|
| `01_quickstart.py` | the whole analysis in a dozen lines | a small file in `tests/data` |
| `02_full_workflow.py` | a real analysis end to end, with figures and marker genes | PBMC 3k, downloaded once (about 8 MB), or your own file |
| `03_large_data_out_of_core.py` | budgets, `ps.plan()` and the out-of-core mode | any `.h5ad`, given as an argument |
| `04_plotting.py` | every plot type, themes and multi-panel figures | PBMC 3k or your own file |
| `05_comparing_conditions.py` | control against treated: two runs read as one, doublets, integration and a per-cell-type comparison with donors as replicates | Kang et al. 2018 from GEO (about 60 MB, once); needs `pip install anndata` |
| `06_large_run.py` | one large dataset (one file or many) end to end, with a run report | your files; with none it runs on the small test file |

```bash
pip install polariseq
git clone https://github.com/DarkOxygen123/Polariseq.git polariseq
cd polariseq
python examples/01_quickstart.py
python examples/02_full_workflow.py
python examples/03_large_data_out_of_core.py /path/to/your.h5ad
python examples/04_plotting.py
python examples/05_comparing_conditions.py
python examples/06_large_run.py
```

Figures go to `examples/output/`, downloads to `examples/data/`.

## Your own data

`02` and `04` read `INPUT` with the right reader for its type: a `.h5ad`
file, a 10x Genomics `.h5` file, or a 10x matrix folder (matrix.mtx,
barcodes.tsv and genes.tsv or features.tsv). `03` takes a `.h5ad` path as an
argument. `06` takes a file, a folder of `.h5ad` files or a pattern such as
`"data/part_*.h5ad"`, in its settings block or on the command line.

## A very large run, on another machine

`06_large_run.py` is the script for one dataset of millions of cells kept as
many `.h5ad` files. Copy it, edit the SETTINGS block at its top (the input
folder, the output and project folders, memory, threads, the analysis
parameters) and run it; every setting can also be passed on the command line.
It prints what the run would need in memory and on disk before anything runs,
then runs the standard workflow (PCA on the selected genes) with every step
timed, a line in the log as each step starts and ends, and the memory held,
the memory free and the disk written sampled twice a second, and leaves in
`--out`:

| file | |
|---|---|
| `run_summary.md` | what was run, on what machine (processor, cores, memory, storage), the time and memory of each step and the disk used: ready for a methods section |
| `report.json` | the same as numbers, with the plan |
| `resources.csv`, `resources.png` | memory, free memory and disk against time, every step shaded |
| `qc.png`, `pca.png`, `clusters.png`, `umap.png` | the figures, drawn from every cell at any size by `ps.pl.overview` (above a million cells the UMAP is drawn as density); `qc.png` shows the cells the quality filter removed, shaded and counted |
| `cells.tsv.gz` | one row per cell kept: name, file, genes, counts, cluster, UMAP, and the input columns named with `--keep-obs` |

```bash
python examples/06_large_run.py /data/atlas.h5ad --out runs/atlas \
    --project /scratch/$USER/polariseq-atlas --ram-gb 64 --threads 16
```

To get Polariseq running on a node where `pip install polariseq` is not an
option (clone, virtual environment, build, or an offline wheel), to size a run,
to run it under a scheduler and to read the logs, see
[`NODE_SETUP.md`](NODE_SETUP.md). `setup_node.sh` does the setup in one command.
`_resources.py`, the small time-and-memory tracker the script uses, works
around any code of yours.

## Two things worth knowing before you copy code out of these

**`min_genes` is a property of your data, not a constant.** 200 is the usual
choice for a droplet dataset with ~30,000 genes. The bundled fixture has
2,000 genes, where 200 would discard 98 % of it. `01` picks a threshold from
the QC metrics rather than hard-coding one; do the same.

**Pin the budget for anything you want to reproduce.** Left alone,
`ps.budget()` follows whatever RAM is free, and the block size follows the
budget — so two runs on one machine can disagree. `ps.set_budget(ram_gb=...)`
makes them identical.

## Data used by the examples

`05_comparing_conditions.py` uses the interferon-beta experiment of Kang, H. M.,
Subramaniam, M., Targ, S. et al. Multiplexed droplet single-cell
RNA-sequencing using natural genetic variation. *Nature Biotechnology* 36,
89-94 (2018), downloaded from the authors' deposit at GEO (GSE96583, batch 2):
peripheral blood mononuclear cells of eight donors, cultured without and with
interferon-beta, with the authors' donor, cell type and doublet annotations.

`02_full_workflow.py` and `04_plotting.py` use the 10x Genomics "3k PBMCs from
a Healthy Donor" dataset (Cell Ranger 1.1.0), downloaded from 10x Genomics.
