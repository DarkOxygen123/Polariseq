# Results

The measured tables of the Polariseq benchmarks, exported from the scripts'
raw output (one JSON record per run, kept out of git for size). Machine
details, the data and the commands are in [../REPRODUCE.md](../REPRODUCE.md).

Dataset names used throughout: `pbmc3k`, `pbmc10k`, `neuron10k`, `pbmc20k`
(public 10x sets), `brain50k` ... `brain1306k` (subsets of the 10x 1.3M mouse
brain file, seed 0), `fetal1000k`, `fetal2000k`, `fetal4062k` (the human fetal
atlas of Cao et al. 2020). Tools: `polariseq-ram` (in memory),
`polariseq-spill` (out of core), `scanpy-default` (Scanpy as shipped),
`scanpy-tuned` (Scanpy configured for speed), `scanpy-tuned-seed1` (the same,
another seed), `scanpy-tuned-int64` (with 64-bit indices, for the 4.06M-cell
file), `seurat-tuned`, `seurat-bpcells`, `bpcells-native`, `scarf`.

Times are in seconds and memory in GB (10^9 bytes) unless a column says
otherwise.

## laptop/: MacBook Air M2, 8 GB, 4 threads

| file | rows | what | from |
|---|---:|---|---|
| `runs_all_tools.csv` | 281 | every tool on every dataset, no UMAP. Polariseq runs Scanpy's steps here (`workflow` = scanpy), so the tools do the same work | `r3_campaign.py` |
| `runs_polariseq_default.csv` | 103 | Polariseq with its own default workflow, no UMAP | `r4_campaign.py` |
| `runs_with_umap.csv` | 209 | every tool with UMAP (Polariseq on its default workflow) | `r3_campaign.py --umap`, `r4_campaign.py --umap` |

One row per run, written by `benchmarks/export_tables.py`. Main columns:
`status` (ok, or why the run stopped: out of memory, time limit), `rep`, the
settings (`threads`, `budget_gb`, `n_hvg`, `n_pcs`, `n_neighbors`,
`resolution`, `min_genes`, `seed`), `wall_s` and `pipeline_s` (the analysis
without process start-up and imports: the times quoted for each tool), the time
of each step (`load_s` ... `umap_s`; the out-of-core runs make one call from
quality control through clustering, timed as `preprocess_s`),
`peak_footprint_gb` (the headline memory measure on macOS), `peak_rss_gb`,
`min_free_gb`, and what came out (`cells_kept`, `genes_kept`, `clusters`).

## planner/

| file | what | from |
|---|---|---|
| `budget_sweep_brain800k.csv` | Polariseq out of core on 800,000 brain cells with budgets of 1, 2, 3, 4, 6 and 8 GB. The planner declines the budgets below its floor (status failed). The others show whether the peak stays inside the budget | `r3_analyses.py budget` |

## node/: 2 x Xeon Platinum 8268, 4 threads, 100 GB budget

| file | what | from |
|---|---|---|
| `runs.csv` | one row per run on `brain1306k` and `fetal4062k`: each tool's time from quality control to clusters, its UMAP time, peak memory, the disk its store used, clusters, the node it ran on (`node`, A to H, all of one type) and that node's load when the run started | `slurm/node_run.sbatch` |
| `steps.csv` | the start, end and duration of every step of those runs | the same records |
| `memory_over_time.csv.gz` | memory held by each run over time, with the step it was in | the same records |
| `agreement.csv` | how close the tools' outputs are on the same data: cells kept, clusters, shared selected genes, principal components, and cluster agreement (adjusted Rand index) between each pair | the same runs |
| `node.json` | the node: processor, cores, memory, storage, operating system, scheduler | |
| `step_resolved/<dataset>/` | Polariseq out of core, step by step, from `examples/06_large_run.py`: `run_summary.md` (readable), `report.json` (the same as numbers), `resources.csv` (memory and disk, twice a second) | `examples/06_large_run.py --mode out-of-core --threads 4` |

## atlas_22m/: the 22.2 million cell scTab atlas, 8 threads

| file | what |
|---|---|
| `run_summary.md`, `report.json` | the run: data, settings, machine, the time and peak memory of each step, disk used |
| `resources.csv` | memory held, memory free and disk free, twice a second, with the step |
| `overview_data.json` | the numbers behind the overview figures: quality control, cluster sizes, variance explained per principal component |
| `cell_types_summary.json` | the clusters against scTab's published cell types: cells, clusters, published types and tissues, types recovered |

Made by `slurm/large_run.sbatch` (`examples/06_large_run.py`). The per-cell
table (`cells.tsv.gz`, 290 MB) is not included.

## cell_types/: clusters against the published cell types of the fetal atlas

| path | what | from |
|---|---|---|
| `fetal_all_cells_res<r>/` | all 4.06M cells analyzed together at Leiden resolution 0.5, 1, 2, 3 and 4: `summary.json` (run, time), `comparison.json` (adjusted Rand index, normalized mutual information, types recovered, rare types), `per_type.csv` (each published type's best cluster, recall and purity), `marker_agreement.csv` (each cluster's markers against its type's published markers) | `fetal_biology.py` |
| `fetal_per_organ/` | the same comparison with each organ analyzed on its own, as the authors did | `fetal_matched.py organs` and `compare` |
| `scanpy_per_organ/` | Scanpy running the authors' per-organ procedure, and Polariseq on the same organs | `r3_analyses.py scanpy_organs` |
| `sampling/summary.json` | what analyzing a random or geometric sketch (1% to 25% of the cells) loses against all cells | `r3_analyses.py subsample` |
| `paris/` | hierarchical clustering (Paris) of the 4.06M-cell graph: time, memory and the cuts against the published types (`fetal_paris.json`), and the tree (`tree.json`) | `validate_paris.py fetal` |
| `tools_fetal_atlas.json` | every tool's clusters and UMAP of the 4.06M cells on the node, measured one way: agreement with the published types, islands in the layout, pixel purity, global structure | `r4_tools_fetal.py` |

A published type counts as recovered when one cluster holds at least half of
its cells and is at least half made of it.

## embedding/

| file | what | from |
|---|---|---|
| `umap_vs_umap_learn.json` | Polariseq's UMAP against umap-learn's on the same graph | `r3_analyses.py umap` |
| `fetal_balanced_sample_summary.json` | the settings of the authors' balanced-sample embedding, as repeated | `fetal_matched.py embed` |
| `multilevel_start_convergence.json` | how close each UMAP start comes to the graph's leading eigenvectors, and its time | `r4_convergence.py` |
| `umap_settings_sweep.csv` | UMAP start, minimum distance and epochs on several pipelines: time, memory, islands, types kept in one piece, pixel purity, global structure | `r4_sweep.py` |

## gene_selection/

| file | what | from |
|---|---|---|
| `pipelines_sweep.csv` | gene selection and normalization methods: time and memory per step, seed-to-seed Leiden agreement, and agreement with the published types at resolutions 1 and 0.5 | `r4_sweep.py` |
| `fetal_gene_detection.csv.gz` | every gene of the fetal atlas: in how many cells it is detected and which pipelines select it | `r4_sweep.py` |
| `genes_selected_<tool>.txt` | the 2,000 genes each tool selected on the 4.06M cells | `r4_tools_fetal.py` |

## Other analyses

| file | what | from |
|---|---|---|
| `downstream/paga_handoff.json` | Polariseq's analysis of 4.06M cells handed to Scanpy's PAGA: time and memory of each stage, and the PAGA groups with their main published type and organ | `r3_downstream.py` |
| `pbmc3k_tutorial/comparison.json` | Scanpy's PBMC 3k tutorial reproduced (cells, clusters, markers, cell-type names), then compared with Polariseq on its default workflow and with Scanpy running Polariseq's steps | `pbmc3k_tutorial.py` |
| `conditions_kang/results.json` | comparing conditions on Kang et al. 2018: doublets against the authors' genotype calls, the joint analysis, significant genes, interferon-stimulated genes per cell type | `r3_kang.py` |
| `conditions_kang/limma_agreement.json` | Polariseq's pseudobulk test against edgeR and limma on the same sums: the largest relative difference in fold change, t statistic and p-values | `r3_kang_agreement.py` |
