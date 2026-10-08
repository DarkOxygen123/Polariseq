# Run summary

Polariseq 0.1.0b1, 2026-10-05 06:30, mode: out-of-core.

## Data

- 22,190,622 cells, 19,331 genes and 46,187,527,591 stored values in 100 file(s)
- after quality control: 22,137,595 cells and 19,307 genes; 2,000 highly variable genes used for PCA; 97 clusters

## Settings

- cells kept with at least 200 genes; genes kept in at least 3 cells; counts scaled to 10,000 per cell, then log1p
- 2000 highly variable genes (Seurat flavour), 50 principal components, 15 neighbours, Leiden resolution 1.0, UMAP yes, seed 0

## Machine

- processor: Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz
- cores: 48 physical, 48 logical; the run used 8 threads
- memory: 201.5 GB; Polariseq's budget 100 GB
- scheduler allocation: 8 cores, 154 GB
- project storage: lustre, 698984.7 GB free at the start
- operating system: Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.17; Python 3.12.14

## Time and memory

| step | time | peak memory held (GB) |
|---|---:|---:|
| open | 27.3 s | 2.8 |
| qc (and the one-time copy of the counts) | 10.3 min | 6.6 |
| normalize | 5.3 min | 4.3 |
| hvg | 9.3 min | 39.4 |
| renormalize | 5.0 min | 4.1 |
| pca | 29.0 min | 28.0 |
| neighbors | 14.3 min | 30.8 |
| leiden | 10.3 min | 27.0 |
| umap | 19.8 min | 33.9 |
| figures | 29.6 s | 12.6 |
| export | 5.5 min | 10.9 |
| **total** | **1.83 h** | **39.4** |

Lowest free memory on the machine during the run: 142.9 GB.
Project folder at the end 58.2 GB; written during the run 65.5 GB.

## Outputs

cells.tsv.gz, clusters.png, figure_data.json, pca.png, qc.png, report.json, resources.csv, resources.png, run_summary.md, umap.png, umap_cell_type.png
