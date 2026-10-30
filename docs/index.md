---
hide:
  - navigation
---

# Polariseq { .home-title }

![Polariseq. Atlas-scale single-cell analysis on the computer you have.](assets/banner-light.svg#only-light)
![Polariseq. Atlas-scale single-cell analysis on the computer you have.](assets/banner-dark.svg#only-dark)

Polariseq analyzes single-cell RNA-seq data from raw counts to clusters, UMAP, marker genes and comparisons
between conditions. It is a Python package with a Rust core that plans each step's memory and disk before it
reads, then runs in memory or out of core with the same code. A few million cells fit on an everyday
computer and tens of millions on one server node.

[Install](install.md){ .md-button .md-button--primary } [Quick start](QUICKSTART.md){ .md-button } [API reference](api/index.md){ .md-button }

## Highlights

<div class="grid cards" markdown>

-   **Plans before it reads**

    ---

    `ps.plan` predicts the memory and disk of every step from the file's description; `ps.set_budget`
    holds the run to them.

-   **One workflow, any size**

    ---

    The same code runs in memory or out of core, where the counts sit in a lossless store of about one
    byte per value.

-   **Repeatable results**

    ---

    The neighbor graph and Leiden clusters are the same at any thread count for a given seed; UMAP
    layouts can be bit-reproducible.

-   **Several samples**

    ---

    Doublet detection, Harmony integration and comparisons between conditions with donors as replicates.

-   **Cell types**

    ---

    `ps.tl.auto_annotate` names clusters from marker databases, with a score and Cell Ontology terms.

-   **Part of the ecosystem**

    ---

    Results export to `.h5ad` and AnnData for Scanpy, scVI and the tools built on them.

</div>

How each step works: [Methods](METHODS.md).

## An atlas on a laptop

![UMAP of the 4.06 million cells of the human fetal atlas, colored by cell type](assets/hero_light.png#only-light)
![UMAP of the 4.06 million cells of the human fetal atlas, colored by cell type](assets/hero_dark.png#only-dark)

| dataset | cells | machine | time | peak memory |
|---|---:|---|---:|---:|
| mouse brain (10x Genomics) | 1.3 million | laptop, 8 GB, 4 threads | 3.6 min | 3.2 GB |
| human fetal atlas (Cao et al. 2020) | 4.06 million | laptop, 8 GB, 4 threads | 5.9 min | 4.2 GB |

<small>Quality control through clustering, out of core, default settings, 6 GB budget, median of 3 runs.
Peak resident memory; the macOS physical footprint peaked at 3.9 GB and 5.7 GB. Scripts and results are on
the [`reproduce`](https://github.com/DarkOxygen123/Polariseq/tree/reproduce) branch.</small>

## Install

```bash
pip install "polariseq[plot] @ git+https://github.com/DarkOxygen123/Polariseq.git"
```

Python 3.10 to 3.14 on macOS, Linux or Windows. `[plot]` adds the figures. The core alone needs only
NumPy and pandas, with no `anndata` package. Each operating system, the extras, clusters and offline
machines: [Installation](install.md).

## Quick start

```python
import polariseq as ps

ps.project("my_analysis")           # folder for everything Polariseq writes
ps.set_budget(ram_gb=6, threads=4)  # the memory and threads it may use

ps.plan("data.h5ad")                # predicted memory and disk, before reading
adata = ps.read_h5ad("data.h5ad")   # adata.mode says where the analysis runs

ps.pp.calculate_qc_metrics(adata)
ps.pp.filter_cells(adata, min_genes=200)
ps.pp.filter_genes(adata, min_cells=3)
ps.pp.normalize_total(adata, target_sum=1e4)
ps.pp.log1p(adata)
ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
ps.pp.renormalize(adata)            # each cell normalized again over the selected genes
ps.pp.pca(adata, n_components=50, scale=True)
ps.pp.neighbors(adata, n_neighbors=15)
ps.tl.leiden(adata, resolution=1.0)
ps.tl.umap(adata)

ps.tl.rank_genes_groups(adata, groupby="leiden")
print(ps.get.markers(adata, n=5))   # top marker genes of each cluster
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")
```

## Citation

If you use Polariseq, or results produced with it, cite it with the version you used:

> Uthamkumar M, Dutta A, Basak T. Polariseq: atlas-scale single-cell analysis on the computer you have.
> Version 0.2.2 (2026). https://github.com/DarkOxygen123/Polariseq

## License

Polariseq is free for noncommercial use (academic research, teaching, universities, research institutes and
nonprofit organizations) under the PolyForm Noncommercial License 1.0.0 with additional terms. Commercial
use requires an agreement with the authors. Contact: trayambak@iitmandi.ac.in.
