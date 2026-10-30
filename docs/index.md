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

```bash
pip install "polariseq[plot] @ git+https://github.com/DarkOxygen123/Polariseq.git"
```

Python 3.10 to 3.14 on macOS, Linux or Windows. `[plot]` adds the figures, and
[Installation](install.md) lists the other extras.

## Where to start

<div class="grid cards" markdown>

-   **[Getting started](install.md)**

    ---

    Install Polariseq, then run [a first analysis](QUICKSTART.md) from a file to clusters and marker
    genes. The [notebook](getting-started/every-command.ipynb) runs every command once.

-   **[How-to guides](how-to/index.md)**

    ---

    One task each: data larger than memory, doublets, batches, comparing conditions, cell types,
    figures, saving and export.

-   **[API reference](api/index.md)**

    ---

    Every function of `ps`, `ps.pp`, `ps.tl`, `ps.get` and `ps.pl`, with its arguments, defaults and an
    example.

-   **[Background](background/index.md)**

    ---

    How Polariseq works: the analysis object, memory and planning, out-of-core analysis, and
    reproducible results. [Methods](METHODS.md) gives the algorithms and their sources.

-   **[Rust](rust/index.md)**

    ---

    The Rust core used directly, with the crates' API reference.

-   **[Source on GitHub](https://github.com/DarkOxygen123/Polariseq)**

    ---

    The code, the [example scripts](how-to/examples/index.md), issues, and the
    [`reproduce`](https://github.com/DarkOxygen123/Polariseq/tree/reproduce) branch with the benchmarks.

</div>

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

## Citation

If you use Polariseq, or results produced with it, cite it with the version you used:

> Uthamkumar M, Dutta A, Basak T. Polariseq: atlas-scale single-cell analysis on the computer you have.
> Version 0.2.2 (2026). https://github.com/DarkOxygen123/Polariseq

## License

Polariseq is free for noncommercial use (academic research, teaching, universities, research institutes and
nonprofit organizations) under the PolyForm Noncommercial License 1.0.0 with additional terms. Commercial
use requires an agreement with the authors. Contact: trayambak@iitmandi.ac.in.
