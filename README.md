<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/banner-dark.svg">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/banner-light.svg" alt="Polariseq. Atlas-scale single-cell analysis on the computer you have." width="100%">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.14-3E7FC1?labelColor=2E3340" alt="Python 3.10 to 3.14">
  <img src="https://img.shields.io/badge/platforms-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-3E7FC1?labelColor=2E3340" alt="macOS, Linux and Windows">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-3E7FC1?labelColor=2E3340" alt="PolyForm Noncommercial 1.0.0"></a>
  <a href="https://github.com/DarkOxygen123/Polariseq/actions"><img src="https://img.shields.io/github/actions/workflow/status/DarkOxygen123/Polariseq/ci.yml?branch=main&label=CI&labelColor=2E3340" alt="CI status"></a>
  <a href="https://darkoxygen123.github.io/Polariseq/"><img src="https://img.shields.io/badge/docs-darkoxygen123.github.io-3E7FC1?labelColor=2E3340" alt="Documentation"></a>
</p>

<p align="center">
  <a href="#install"><b>Install</b></a> &nbsp;·&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="https://darkoxygen123.github.io/Polariseq/"><b>Documentation</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/DarkOxygen123/Polariseq/tree/reproduce"><b>Reproduce</b></a>
</p>

Polariseq analyzes single-cell RNA-seq data from raw counts to clusters, UMAP, marker genes and comparisons
between conditions. It is a Python package with a Rust core that plans each step's memory and disk before it
reads, then runs in memory or out of core with the same code. A few million cells fit on an everyday
computer and tens of millions on one server node.

## Highlights

<table>
<tr>
<td width="33%" valign="top"><b>Plans before it reads</b><br><code>ps.plan</code> predicts the memory and disk of every step from the file's description; <code>ps.set_budget</code> holds the run to them.</td>
<td width="33%" valign="top"><b>One workflow, any size</b><br>The same code runs in memory or out of core, where the counts sit in a lossless store of about one byte per value.</td>
<td width="33%" valign="top"><b>Repeatable results</b><br>The neighbor graph and Leiden clusters are the same at any thread count for a given seed; UMAP layouts can be bit-reproducible.</td>
</tr>
<tr>
<td valign="top"><b>Several samples</b><br>Doublet detection, Harmony integration and comparisons between conditions with donors as replicates.</td>
<td valign="top"><b>Cell types</b><br><code>ps.tl.auto_annotate</code> names clusters from marker databases, with a score and Cell Ontology terms.</td>
<td valign="top"><b>Part of the ecosystem</b><br>Results export to <code>.h5ad</code> and AnnData for Scanpy, scVI and the tools built on them.</td>
</tr>
</table>

How each step works: [Methods](https://darkoxygen123.github.io/Polariseq/METHODS/).

## An atlas on a laptop

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/hero_dark.png">
    <img src="https://raw.githubusercontent.com/DarkOxygen123/Polariseq/main/docs/assets/hero_light.png" alt="UMAP of the 4.06 million cells of the human fetal atlas, coloured by cell type, with the laptop time and memory" width="100%">
  </picture>
</p>

| dataset | cells | machine | time | peak memory |
|---|---:|---|---:|---:|
| mouse brain (10x Genomics) | 1.3 million | laptop, 8 GB, 4 threads | 3.6 min | 3.2 GB |
| human fetal atlas (Cao et al. 2020) | 4.06 million | laptop, 8 GB, 4 threads | 5.9 min | 4.2 GB |

<sub>Quality control through clustering, out of core, default settings, 6 GB budget, median of 3 runs.
Peak resident memory; the macOS physical footprint peaked at 3.9 GB and 5.7 GB. Scripts and results are on
the <a href="https://github.com/DarkOxygen123/Polariseq/tree/reproduce"><code>reproduce</code></a> branch.</sub>

## Install

```bash
pip install "polariseq[plot] @ git+https://github.com/DarkOxygen123/Polariseq.git"
```

Python 3.10 to 3.14 on macOS, Linux or Windows. pip compiles the Rust core and fetches what the build needs;
a C compiler is the one tool to have (macOS: `xcode-select --install`; Windows: Build Tools for Visual
Studio; Linux: usually present). `[plot]` adds the figures; the core alone needs only NumPy and pandas, with no `anndata` package.
Other extras, each operating system, clusters and offline machines: [Installation](https://darkoxygen123.github.io/Polariseq/install/).

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

## Documentation

The documentation is at **https://darkoxygen123.github.io/Polariseq/**.

| | |
|---|---|
| [Installation](https://darkoxygen123.github.io/Polariseq/install/) | each operating system, the extras, cluster nodes, offline machines, building from source |
| [A first analysis](https://darkoxygen123.github.io/Polariseq/QUICKSTART/) | from a file to clusters and marker genes, step by step |
| [How-to guides](https://darkoxygen123.github.io/Polariseq/how-to/) | one task each: data larger than memory, doublets, batches, conditions, cell types, figures, export |
| [Example scripts](https://darkoxygen123.github.io/Polariseq/how-to/examples/) | seven runnable scripts, from a first analysis to a run of millions of cells |
| [API reference](https://darkoxygen123.github.io/Polariseq/api/) | every function with its arguments, defaults and an example |
| [Background](https://darkoxygen123.github.io/Polariseq/background/) | the analysis object, memory and planning, out-of-core analysis, reproducibility |
| [Methods](https://darkoxygen123.github.io/Polariseq/METHODS/) | the algorithms, the work they build on, and their references |
| [Rust crates](https://darkoxygen123.github.io/Polariseq/rust/) | the Rust core used directly, with its API reference |
| [Reproduce](https://github.com/DarkOxygen123/Polariseq/tree/reproduce) | the benchmark scripts and their results |

Questions and bugs: [open an issue](https://github.com/DarkOxygen123/Polariseq/issues) with what you ran,
what happened and the output of `python -c "import polariseq as ps; print(ps.get_build_info())"`.

## Citation

If you use Polariseq, or results produced with it, cite it with the version you used. The citation is in
[CITATION.cff](CITATION.cff), and GitHub's "Cite this repository" button gives it in APA and BibTeX:

> Uthamkumar M, Dutta A, Basak T. Polariseq: atlas-scale single-cell analysis on the computer you have.
> Version 0.2.2 (2026). https://github.com/DarkOxygen123/Polariseq

## License

Polariseq is released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
with additional terms, all in [LICENSE](LICENSE):

- **Free for noncommercial use**: academic research, teaching, and use by universities, research
  institutes and nonprofit organizations, whatever their funding, as long as the work is not a paid
  service or product for a company.
- **Keep the notices**: copies and changed versions carry [LICENSE](LICENSE) and
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and a changed version says it is based on Polariseq.
- **Cite it**: work that uses Polariseq cites it (see [Citation](#citation)).
- **Commercial use requires an agreement with the authors**, including use inside a company and analyses
  run as a paid service. Contact: trayambak@iitmandi.ac.in.

Some steps follow code published by others under permissive licences; those projects and their licences
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (C) 2026 Uthamkumar M, Abhi Dutta and Trayambak Basak.
