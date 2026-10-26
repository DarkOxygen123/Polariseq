# Changelog

## Unreleased

- **No spurious thread warning.** An analysis that ran parallel work before reading the budget (an
  object made from an array, then `pp.pca`) warned "could not set the worker pool to 4 threads" on
  the first step that read it, though no thread count had been asked for. The budget now takes the
  thread count the worker pool really has, silently; the warning stays for a count asked for with
  `set_budget(threads=...)` after the pool has started, the one case where a setting is not
  honoured.

## 0.2.2 (2026-10-18)

- **Cell type annotation, rewritten.** `ps.tl.auto_annotate` scores every cell for every cell type
  with the published ScType and Cell Marker Accordion methods (markers standardized over the cells,
  weighted by how many sources name them and how few types share them, genes a type should not
  express counted against it), in the Rust core in two streamed passes, in memory or out of core. It
  reads every gene of the source files, so markers outside the selected genes count. It returns a
  table per cluster: the name and its Cell Ontology term, the score, the share of the cluster's cells
  supporting it, the runner-up, and groups of cells inside the cluster scoring for another type (rare
  populations); every cell also gets its own best type. A cluster no type fits stays `unassigned`;
  two types scoring within 5% of each other give their shared parent type. `source=` chooses the
  markers: shipped, Polariseq's curated list and the Human Protein Atlas single cell types (the
  default, both), and MSigDB C8; fetched once on request, the Cell Marker Accordion, CellMarker 2.0,
  PanglaoDB and ScType (`pip install 'polariseq[markers]'`); your own with `markers=` (a dict, a
  table or a `.gmt`/`.csv` file, negative markers allowed). `tissue=` narrows to a tissue, keeping
  blood and immune types; mouse data are mapped through MGI orthologs. `ps.marker_table` shows a
  source's markers. Against published labels (PBMC 3k, Kang et al. 2018, the Cao et al. 2020 fetal
  atlas organ by organ, scored on the Cell Ontology), the new default names 73% of cells in the
  right lineage on average over the three (77% with `tissue=`), against 34% for 0.2.0's PanglaoDB
  scoring, and 51% of the cells of rare types (under 1% of an organ's cells) against 24%; with the
  Accordion added (`source=("curated", "hpa", "accordion")`), 80% and 69%
  (`benchmarks/annotation/annot_bench.py`). PanglaoDB is no longer shipped (`source="panglaodb"`
  fetches it); `method=` is still read.

## 0.2.1 (2026-10-17)

- **Saved analyses.** `adata.save(name)` keeps an analysis in the project folder
  (`states/<name>/`) and `ps.load(project, name)` opens it again where it stopped. The counts are
  pointed to, not copied: out of core the on-disk copy is reopened with the cells, genes and
  normalization the steps left; in memory the matrix is rebuilt from the source files by the same
  steps, identical to the last bit (checked at save and at load). Results the project holds are hard
  links. A moved source file stops the load with `ps.SourceMovedError`, naming where it was expected
  (`relink=` gives its new place); paths inside the project are kept relative, so a project moved
  with its data keeps loading. The format is typed JSON plus `.npy` files: loading runs no code.
- **Several analyses of one dataset.** `adata.from_counts(name)` starts a new analysis of the same cells
  from their raw counts (out of core a new view of the same on-disk copy, nothing copied; in memory the
  cells read again from the source files), keeping the cells' own annotations (quality metrics, doublet
  scores, columns you added) and leaving what the first analysis derived (clusters, scores, labels,
  embeddings). Each analysis keeps its results in `runs/<dataset>/analyses/<name>/`, so analyses never
  replace each other's files; run with the same settings, a new analysis reproduces the first exactly.
- **Comparing analyses.** `ps.tl.compare_analyses({name: analysis, ...}, labels=..., seeds=...)` measures
  how far analyses of the same cells agree: adjusted Rand index and normalized mutual information for
  every pair, which cluster became which (cells and top markers shared), agreement with given labels,
  and, with `seeds`, each analysis's agreement with itself under other seeds, the run-to-run variation a
  difference between settings should exceed. `ps.pl.compare_analyses` draws each analysis's clusters
  side by side on one embedding.
- **Documentation.** `docs/API.md` lists every public function with its arguments, their types,
  defaults and meaning, and what it returns, written from the code (`python -m polariseq._api_reference`;
  a test fails when it falls behind or an argument is undescribed). `notebooks/02_every_command.ipynb`
  runs every command once on PBMC 3k; `examples/07_settings_saving_exporting.py` shows several analyses
  with different settings, their agreement, saving, reopening and export. Arguments that had no
  description in `help()` now have one (`tl.leiden`, `get.pseudobulk`, the plots, ...).
- **Plots: a validated palette, dark mode done properly, and nothing collides.** The group palette
  is eight hues checked as a set for each surface (lightness band, colour strength, colour-blind and
  normal-vision separation of neighbours, contrast), with steps of its own for dark backgrounds;
  0.2.0's had a grey slot that read as "no group", red and pink side by side, and a navy that
  vanished on dark. Past eight groups the hues repeat lighter, then darker, and the groups are
  numbered on the plot (a cluster keeps its own number). Values take a one-hue blue scale whose low
  end stays visible, signed values blue to red through grey, each with a dark version where low
  values recede instead of glaring white. Grey text reaches 4.5:1 on every background. Fixed:
  legends and gene or group names drawn black on the dark theme; `legend_loc="on data"` drawing no
  names (they are now placed so none overlaps another); y-axis labels colliding with subtitles;
  ticks colliding in the notebook and talk themes (figures now grow with the type); panels of a
  multi-panel embedding at different heights; log axes reading 6x10^0; violin names overlapping and
  only the first panel labelled; "not tested" and "no change" sharing white in `fold_changes`
  (untested cells are hatched). `ps.pl.set_theme("classic")` keeps 0.2.0's look; the scripts of the
  paper's and the README's figures pin it.
- **Numbered groups in numeric order in plots** (clusters "0" to "20" read 0, 1, 2, ... 10, not 0, 1,
  10, ...), also when the labels are text, as after an export or a load.
- **`adata.write_h5ad(path)`** writes the anndata format natively (no `anndata` package needed): X as
  the analysis left it (normalized, selected genes), streamed from the on-disk copy out of core, and
  every annotation, embedding, graph and result with Scanpy's conventions. `uns['polariseq']` points to
  the source files and records which of their cells and genes the file holds and the steps that made X;
  `counts=True` also writes the raw counts as `layers['counts']`. Written under a temporary name and
  renamed when complete.
- **Genes of your own in the gene selection.** `ps.pp.highly_variable_genes(..., include=[...])` (and
  `process_diskbacked(include=...)`) keeps the genes named whatever the ranking chooses: those not among
  the top `n_top_genes` are added (past the detection floor), so renormalization and PCA run on them too.
  `var['included']` marks them and `uns['hvg']` records the request. A name not in the data is an error
  saying, per gene, whether it is absent or was removed by an earlier step, with the closest names; one
  detected in very few cells is kept with a warning.
- **Errors that say what is wrong and what to do.** A misspelled column, gene, group or embedding names
  the closest match and what exists; a step run before the one it needs names that step (`neighbors`
  before `pca`, `leiden` before `neighbors`, ...); a missing, folder or non-HDF5 input says what is there
  and which reader fits; impossible values (`set_budget(threads=0)`, `pca` with more components than the
  data allow, a `dpt` root that is not a cell) are refused with the limit. Some mistakes passed silently
  before and are now errors: unknown groups in `rank_genes_groups(groups=...)` and `pl.volcano`, and
  non-positive budgets; genes missing from `score_genes` or `pl.dotplot` are named in a warning. Missing
  names raise `KeyError` as before, printed as a plain sentence.
- **Export after Scrublet.** `to_anndata()` hands memory-mapped results (the simulated doublets'
  scores, kept in the project) to anndata as plain arrays, so `write_h5ad` works.
- **Gene selection with `subset=True` records the genes it keeps**, as the gene filter does. Before,
  steps that read the raw counts back from the files after it (`compare_conditions`,
  `get.pseudobulk` on a normalized matrix in memory) took the genes before the selection.

## 0.2.0 (2026-10-16)

**Licence.** From 0.2.0, Polariseq is released under the PolyForm Noncommercial License 1.0.0 with
additional terms (keep the notices, cite Polariseq; commercial use by agreement with the authors), in
`LICENSE`. Versions 0.1.0 and 0.1.1 were released under GPL-3.0-or-later and are withdrawn. Polariseq is
installed from this repository (`pip install "git+https://github.com/DarkOxygen123/Polariseq.git"`); the
PyPI packages of 0.1.x were removed.

**A light install.** `pip install "git+https://github.com/DarkOxygen123/Polariseq.git"` installs the
analysis with NumPy and pandas only. Figures (`plot`, Matplotlib), SciPy matrices for other tools (`scipy`),
AnnData export (`anndata`) and CellTypist annotation (`celltypist`) are extras, and
`all` adds every one. Without SciPy the neighbour graph in `adata.obsp` holds its CSR arrays and gives the
same clusters and layouts. pip fetches CMake for the build, and maturin installs Rust when it is missing,
so a C compiler is the one build tool to have installed.

Three modules are written anew, so that none of Polariseq follows code under the GNU General Public
License; NN-Descent and the planner change where measurements at 100 million cells showed they had to.
Details and every comparison are in `benchmarks/relicence/README.md` on the reproduce branch.

- **Leiden** follows Traag, Waltman and van Eck (2019) and the authors' CWTS code (MIT). Reduced graphs
  carry node weights as sums, which fixes an error of 0.1.x: every aggregated level counted the internal
  weight of a community twice too often. On nine datasets from 2,700 to 4 million cells, the clusters agree
  with 0.1.x as well as 0.1.x agrees with itself across seeds, at equal or higher modularity, equal or
  better agreement with published labels, and 12 to 25% less time. Louvain (`method="fast"`) gains most
  from the fix (Kang et al.: agreement with the published types 0.42 to 0.51).
- **Pseudobulk testing** is written from the edgeR and limma publications and documentation, its open
  details settled by comparing outputs with edgeR 4.10.5 and limma 3.68.5: on 350 generated datasets the
  same genes pass the filter and p-values agree within 4e-12 (0.1.x: 195 of 200 datasets, 2e-9). The
  special functions follow the NIST DLMF, with libm's `lgamma` and `erfc`. Results on Kang et al. are
  unchanged (6,106 genes at adjusted p < 0.05, as with R).
- **Harmony** follows Korsunsky et al. (2019) and harmony-pytorch (BSD-3), with R harmony's documented
  options. Corrections agree with harmonypy 2.0.2 as closely as harmonypy agrees with itself across
  seeds, at the speed of 0.1.x (four to seven times harmonypy's).
- **NN-Descent** splits subtrees of more than 2^20 points in place, and copies the rows of smaller ones
  as before. Random hyperplanes split unevenly (on the 1.3M-cell brain atlas up to half of all cells
  reached one copy), so one copy per worker thread was the peak of the neighbours step, about 1 KB per
  cell at 16 threads. The graph is identical bit for bit.
- **Planner**: the neighbours step is charged `680 + 1 x threads` bytes per cell above the embedding,
  fitted with 0.2.0 on slices of the Tahoe-100M embedding (20 million cells at 4 to 16 threads, 40 million,
  and all 95.6 million cells). 0.1.1 planned 129 GB for the 95.6-million-cell Tahoe-100M run, which held
  159.5 GB (178.9 GB with mapped files); 0.2.0's step holds 62.6 GB there. The graph steps are refitted
  on every run of the final build (the Mac, Windows and a compute node, 1.3 and 4.0 million cells): inputs
  read into memory count twice in resident memory (the copy, and the pages of the file the step before
  wrote), inputs read in place once; UMAP also counts the embedding, which its multilevel start reads; the
  run's per-cell annotations are charged 64 bytes per cell; UMAP's working memory is 42 bytes per graph
  entry (31 before). The fetal atlas at 6 GB is now planned at 7.3 GB, against 6.3 to 7.3 GB measured
  (5.4 GB planned before); `benchmarks/relicence/planner_v020_check.py` (reproduce branch) sets plan beside measurement.
- **Graph steps read their inputs into memory when they fit.** Neighbours, Leiden and UMAP read the
  embedding and the graphs in random order. 0.1.x read them in place from the project's files, which on a
  cluster's network file system stalled whenever the node dropped the cached pages: on Tahoe-100M the
  neighbours step and Leiden ran at 0.03 to 0.2 cores for hours, and a failed read there kills the process.
  Each step now reads its inputs into memory in one pass when the run, the inputs and the step's working
  memory fit the budget and the memory free at the time (within a Slurm allocation or a cgroup limit);
  read this way, Tahoe-100M's neighbours step takes 1.0 h (3.8 h in the 0.1.1 run) and Leiden 45 min
  (65 min). Otherwise a step reads them in place; when the project is on a network file system it first
  copies them to a local disk, if one has room, and warns if none has. `ps.plan()` applies the same rule,
  shows each graph step's choice and warns about a network project; `adata.uns[step]["inputs"]` and
  example 06's report record the choice made. UMAP builds its layout in memory and writes it once.

## 0.1.0 and 0.1.1 (2026-10)

The first releases. Both were withdrawn because of an issue found after release, and 0.2.0 replaces them;
their features and fixes are part of 0.2.0.
