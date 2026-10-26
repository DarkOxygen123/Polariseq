# Run summary

Polariseq 0.1.0b1, 2026-10-08 05:35, mode: out-of-core.

## Data

- 4,062,980 cells, 63,561 genes and 2,631,157,163 stored values in 1 file(s)
- after quality control: 4,000,604 cells and 50,171 genes; 2,000 highly variable genes used for PCA; 43 clusters

## Settings

- cells kept with at least 200 genes; genes kept in at least 3 cells; counts scaled to 10,000 per cell, then log1p
- workflow 'polariseq': 2000 highly variable genes (Scanpy's seurat ranking among genes detected in more than 1% of cells); each cell then normalized again over the selected genes (to 1,000, log) and each gene standardized before PCA; 50 principal components, 15 neighbours, Leiden resolution 1.0; UMAP from the multilevel start, minimum distance 0.5, 100 rounds; seed 0

## Machine

- processor: Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz
- cores: 48 physical, 48 logical; the run used 4 threads
- memory: 201.5 GB; Polariseq's budget 100 GB
- scheduler allocation: 4 cores, 66 GB
- project storage: lustre, 698422.7 GB free at the start
- operating system: Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17; Python 3.12.14

## Time and memory

| step | time | peak memory held (GB) |
|---|---:|---:|
| open | 2.2 s | 0.7 |
| qc (and the one-time copy of the counts) | 3.2 min | 2.4 |
| normalize | 17.3 s | 1.0 |
| hvg | 49.7 s | 4.2 |
| renormalize | 69.8 s | 1.2 |
| pca | 2.6 min | 5.5 |
| neighbors | 3.3 min | 4.6 |
| leiden | 1.8 min | 5.3 |
| umap | 5.2 min | 6.2 |
| figures | 8.6 s | 3.4 |
| **total** | **18.5 min** | **6.6** |

Lowest free memory on the machine during the run: 177.0 GB.
Project folder at the end 5.7 GB; written during the run 8.4 GB.

## Outputs

clusters.png, figure_data.json, pca.png, qc.png, report.json, resources.csv, resources.png, run_summary.md, umap.png
