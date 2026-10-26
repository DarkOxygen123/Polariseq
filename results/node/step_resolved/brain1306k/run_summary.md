# Run summary

Polariseq 0.1.0b1, 2026-10-08 05:54, mode: out-of-core.

## Data

- 1,306,127 cells, 27,998 genes and 2,624,828,308 stored values in 1 file(s)
- after quality control: 1,292,537 cells and 22,787 genes; 2,000 highly variable genes used for PCA; 37 clusters

## Settings

- cells kept with at least 200 genes; genes kept in at least 3 cells; counts scaled to 10,000 per cell, then log1p
- workflow 'polariseq': 2000 highly variable genes (Scanpy's seurat ranking among genes detected in more than 1% of cells); each cell then normalized again over the selected genes (to 1,000, log) and each gene standardized before PCA; 50 principal components, 15 neighbours, Leiden resolution 1.0; UMAP from the multilevel start, minimum distance 0.5, 100 rounds; seed 0

## Machine

- processor: Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz
- cores: 48 physical, 48 logical; the run used 4 threads
- memory: 201.5 GB; Polariseq's budget 100 GB
- scheduler allocation: 4 cores, 49 GB
- project storage: lustre, 698417.0 GB free at the start
- operating system: Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17; Python 3.12.14

## Time and memory

| step | time | peak memory held (GB) |
|---|---:|---:|
| open | 0.6 s | 0.3 |
| qc (and the one-time copy of the counts) | 2.6 min | 1.9 |
| normalize | 26.5 s | 0.7 |
| hvg | 46.6 s | 8.2 |
| renormalize | 72.7 s | 1.2 |
| pca | 3.2 min | 2.7 |
| neighbors | 49.8 s | 1.7 |
| leiden | 22.3 s | 1.7 |
| umap | 89.8 s | 1.9 |
| figures | 5.8 s | 1.4 |
| **total** | **11.0 min** | **8.2** |

Lowest free memory on the machine during the run: 173.2 GB.
Project folder at the end 4.7 GB; written during the run 7.8 GB.

## Outputs

clusters.png, figure_data.json, pca.png, qc.png, report.json, resources.csv, resources.png, run_summary.md, umap.png
