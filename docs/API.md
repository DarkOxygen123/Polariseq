# Polariseq API reference

Every public function, with its arguments, their types and defaults, and what
it returns. Written from the code (`python -m polariseq._api_reference`), so it
matches the installed version: 0.2.2.

```python
import polariseq as ps
```

| namespace | what is in it |
|---|---|
| `ps` | the project folder, the memory budget, planning, reading, loading saved analyses |
| the data object | what `ps.read_h5ad` returns: the matrix, annotations, and `save`, `from_counts`, `write_h5ad`, `to_anndata`, subsetting |
| `ps.pp` | preprocessing: quality control, filters, doublets, normalization, gene selection, PCA, integration, the neighbour graph |
| `ps.tl` | tools: clustering, embedding, markers, annotation, curation, trajectories, comparisons |
| `ps.get` | tables taken out of an analysis |
| `ps.pl` | plots (the `plot` extra: `pip install "polariseq[plot]"`) |

## Using it well

- **Declare a project folder first** (`ps.project("my_analysis")`): everything
  Polariseq writes goes there, results are memory-mapped files instead of
  memory, and saved analyses live there.
- **Set the budget** (`ps.set_budget(ram_gb=..., threads=...)`) before the
  first step: it decides in-memory or out-of-core, the block size, and fixes
  the thread count (the worker pool cannot change size later).
- **Ask before running** (`ps.plan(path)`): predicted peak memory and disk, per
  step, from the file's metadata alone.
- **Out of core costs nothing to branch**: `from_counts` gives a new analysis
  of the same cells over the same on-disk copy; compare settings with
  `ps.tl.compare_analyses(..., seeds=2)` and read the result against the
  seed-to-seed agreement it reports.
- **Save, don't export, to continue later** (`adata.save` / `ps.load`): the
  counts are pointed to, not copied. Export (`write_h5ad`, `to_anndata`) when
  another tool needs the data.
- **Name cell types, then check them** (`ps.tl.auto_annotate(adata, "leiden",
  tissue="immune")`): a table per cluster with the name, its Cell Ontology
  term, a score, the share of cells supporting it, the runner-up and rare
  groups inside the cluster. `source=` picks the marker database
  (`ps.marker_table` shows one); give `tissue` when you know it.
- **Errors name the fix**: a misspelled column, gene or file names the closest
  match; a step run too early names the step to run first.

## `ps`: project, budget, planning, reading

Declare the project folder and the budget first; read data with `read_h5ad` (or the 10x readers);
`plan` predicts the cost before anything is read; `load` reopens a saved analysis; `marker_table`
shows the markers `tl.auto_annotate` scores with.

### `ps.project`

```python
ps.project(path: str | Path) -> Project
```

Declare the project folder of this session, creating it if needed.

Everything Polariseq writes to disk from now on goes into it: the copies of the counts an out-of-core analysis reads, the results of the steps, saved analyses and the scratch files of running steps. Put it on a fast local disk. Its `README.md` says what each part holds and what may be deleted.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| Path | required | The folder (created if missing); `~` is expanded. |

**Returns:** The project. `status()` gives the space each part of it takes.

### `ps.set_budget`

```python
ps.set_budget(*, ram_gb: float | None = None, threads: int | None = None, spill_dir: str | None = None, spill_gb: float | None = None, allow_lossy: bool | None = None) -> Budget
```

Set the memory and threads Polariseq may use.

Set it before the first step. Anything you leave out stays as detected. A pinned `ram_gb` also makes two runs of a script identical out of core, where the block size follows the budget.

| argument | type | default | meaning |
|---|---|---|---|
| `ram_gb` | float \| None | `None` | The most memory, in gigabytes, Polariseq may hold. |
| `threads` | int \| None | `None` | The worker threads of every parallel step. They are set once per Python session. |
| `spill_dir` | str \| None | `None` | The folder for out-of-core scratch files. |
| `spill_gb` | float \| None | `None` | The most space those files may take. |
| `allow_lossy` | bool \| None | `None` | Let the planner trade accuracy for fit. Off by default: Polariseq picks a slower exact plan over a faster approximate one unless asked, and reports any approximation it makes. |

**Returns:** The new budget.

### `ps.budget`

```python
ps.budget() -> Budget
```

The budget in force: the memory and threads Polariseq may use.

Detected the first time it is needed (three quarters of the available memory, and the machine's performance cores), unless `set_budget` set it.

**Returns:** The budget.

### `ps.resources`

```python
ps.resources() -> Resources
```

What this machine has: its memory (total and available), cores and scratch space.

`available_source` says how the available memory was found, since a measured figure and an estimate deserve different trust.

**Returns:** The machine's resources.

### `ps.plan`

```python
ps.plan(path: str | list[str] | dict[str, str], *, genes: str = 'intersection', n_hvg: int = 2000, n_pcs: int = 50, n_neighbors: int = 15, storage: str | None = None, with_umap: bool = True, verbose: bool = True)
```

Predict what an analysis will need, before reading the data.

Reads only the file's description (its numbers of cells, genes and stored values), then predicts the peak memory and disk of each step under the budget, names the step that needs the most, and chooses where the analysis would run: in memory when the matrix takes at most half the budget, otherwise out of core. It returns at once, whatever the size of the file.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| list[str] \| dict[str, str] | required | The file; or several, as a list of paths or a dict from each dataset's label to its path, planned as the one dataset `read_h5ad` reads from them. |
| `genes` | str | `'intersection'` | For several files, the genes kept: `"intersection"` (default) or `"union"`. |
| `n_hvg` | int | `2000` | The analysis settings that change the cost. |
| `n_pcs` | int | `50` | The analysis settings that change the cost. |
| `n_neighbors` | int | `15` | The analysis settings that change the cost. |
| `storage` | str \| None | `None` | Plan for a given mode: `"ram"` (in memory) or `"spill"` (out of core). |
| `with_umap` | bool | `True` | Include the UMAP step. |
| `verbose` | bool | `True` | Print the report as well as returning the plan. |

**Returns:** The plan: `fits` (whether it fits the budget), `peak_gb`, `limiting_step`, `rows_per_block`, the cost of every step, and for several files each one's shape and the genes it keeps (`datasets`).

### `ps.max_cells`

```python
ps.max_cells(*, n_genes: int = 30000, nnz_per_cell: int = 2000, storage: str | None = None, **kwargs: Any) -> int
```

The largest number of cells the budget can analyze.

The question `plan` answers, the other way round: how large a dataset of a given density this machine can take, in memory or out of core.

| argument | type | default | meaning |
|---|---|---|---|
| `n_genes` | int | `30000` | The genes in the dataset. |
| `nnz_per_cell` | int | `2000` | Stored (nonzero) values per cell, the density of the data. |
| `storage` | str \| None | `None` | `None` (the planner chooses), `'ram'` (in memory) or `'spill'` (out of core). |

**Returns:** The largest number of cells.

### `ps.read_h5ad`

```python
ps.read_h5ad(path: str | list[str] | dict[str, str], *, sparse: bool = True, lazy: bool = True, mode: str = 'auto', name: str | None = None, batch_key: str = 'dataset', genes: str = 'intersection', obs_columns: list[str] | None = None) -> AnnData | LazyAnnData
```

Open an `.h5ad` file, or several files as one dataset.

Opening reads only the file's description: its cells, genes and annotations. The matrix is read when a step first needs it. The planning step then chooses where the analysis runs from the size of the matrix and the budget: in memory when it fits, otherwise out of core, with the counts copied once to the project folder and read in blocks. `adata.mode` says which; the steps and their results are the same either way.

Several files are analyzed as one dataset: their cells one after another over one set of genes, matched by name, each cell labelled with its file (the file name, or the dict's key) in `obs[batch_key]`. The matrix is the one `anndata.concat` would build, without the files being merged first.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| list[str] \| dict[str, str] | required | The `.h5ad` file; or several, as a list of paths or a dict from each dataset's label to its path. |
| `sparse` | bool | `True` | Keep a sparse matrix sparse (default); `False` reads it as a dense array. |
| `lazy` | bool | `True` | Read the matrix when a step first needs it (default); `False` reads it at once. |
| `mode` | str | `'auto'` | Where the analysis runs: `"auto"` (default) lets the planning step choose, `"memory"` or `"out-of-core"` fixes it. |
| `name` | str \| None | `None` | The dataset's folder in the project (default: the file name without its extension; for several files, their labels joined by `+`). |
| `batch_key` | str | `'dataset'` | For several files, the `obs` column of each cell's file. |
| `genes` | str | `'intersection'` | For several files, the genes kept: `"intersection"` (default; the genes every file has, in the first file's order) or `"union"` (every gene, zero where a file lacks it), ordered as `anndata.concat` orders them. |
| `obs_columns` | list[str] \| None | `None` | The file's cell annotations to read into `obs` (default: all of them). For several files, a column one file lacks is missing for its cells. |

**Returns:** The analysis: a `LazyAnnData` (default), or an `AnnData` with `lazy=False`.

### `ps.read_10x_h5`

```python
ps.read_10x_h5(path: str, *, lazy: bool = True) -> AnnData | LazyAnnData
```

Open a 10x Genomics `.h5` file, as Cell Ranger writes it.

10x files are analyzed in memory. To analyze one out of core, write it to an `.h5ad` file first (`ps.read_10x_h5(path).write_h5ad("data.h5ad")`) and open that.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str | required | The `filtered_feature_bc_matrix.h5` file. |
| `lazy` | bool | `True` | Read the matrix when a step first needs it (default); `False` reads it at once. |

**Returns:** The analysis: a `LazyAnnData` (default), or an `AnnData` with `lazy=False`.

### `ps.read_10x_mtx`

```python
ps.read_10x_mtx(dir: str, *, lazy: bool = True) -> AnnData | LazyAnnData
```

Open a 10x Genomics matrix folder.

The folder holds `matrix.mtx`, `barcodes.tsv` and `features.tsv` (or `genes.tsv`), plain or gzipped. It is analyzed in memory.

| argument | type | default | meaning |
|---|---|---|---|
| `dir` | str | required | The folder. |
| `lazy` | bool | `True` | Read the matrix when a step first needs it (default); `False` reads it at once. |

**Returns:** The analysis: a `LazyAnnData` (default), or an `AnnData` with `lazy=False`.

### `ps.load`

```python
ps.load(project: str | os.PathLike, name: str, *, relink: dict[str, str] | None = None) -> Any
```

Open an analysis saved with `adata.save`, to continue where it stopped.

Declares `project` the project folder, as `project` does, and returns the analysis as it was saved: its cells and genes, the matrix as the steps left it, and every annotation, embedding, graph and result. Loading is quick, because large results are read only when they are used.

| argument | type | default | meaning |
|---|---|---|---|
| `project` | str \| os.PathLike | required | The project folder the analysis was saved in. |
| `name` | str | required | The saved analysis's name. |
| `relink` | dict[str, str] \| None | `None` | New places of source files that have moved, `{old path: new path}`. |

**Returns:** The saved analysis, the same kind of object `read_h5ad` returns.

**Raises:** SourceMovedError: A source file is no longer where it was; the message says where it was expected.

### `ps.marker_table`

```python
ps.marker_table(source: str | list[str] | tuple[str, ...] | None = ('curated', 'hpa'), *, tissue: str | list[str] | None = None, species: str = 'human', markers: Any = None) -> Any
```

The marker genes `tl.auto_annotate` would score with, as a table.

Use it to look at a source before using it, or to start a table of your own: filter it, add rows, and pass it back as `markers=` with `source=None`.

| argument | type | default | meaning |
|---|---|---|---|
| `source` | str \| list[str] \| tuple[str, ...] \| None | `('curated', 'hpa')` | A marker source or a list of them: `"curated"`, `"hpa"`, `"msigdb_c8"` (shipped), `"accordion"`, `"cellmarker"`, `"panglaodb"`, `"sctype"` (downloaded once on first use). `None` with `markers`: only yours. |
| `tissue` | str \| list[str] \| None | `None` | Keep this tissue's types, a name or a list. Immune types and types found in many tissues are kept with any tissue. `None`: all. |
| `species` | str | `'human'` | `"human"` or `"mouse"`; human-only sources are mapped to mouse genes. |
| `markers` | Any | `None` | Your own markers, in any form `auto_annotate` takes. |

**Returns:** A `pandas.DataFrame` with one row per marker: `cell_type`, `gene`, `sign` (+1 expressed, -1 not expressed), `evidence` (how many records name it), `strength` (the source's own ranking value, 0 when it has none), `tissue`, `cl_id` (the Cell Ontology term), `species` and `source`.

### `ps.process_diskbacked`

```python
ps.process_diskbacked(path: str, *, n_hvg: int = 2000, n_pcs: int = 50, n_neighbors: int = 15, target_sum: float = 10000.0, resolution: float = 1.0, seed: int = 42, min_genes: int = 0, max_genes: float = inf, min_counts: float = 0, max_counts: float = inf, max_pct_mt: float = inf, mito_prefix: str = 'MT-', min_cells: int = 0, include: list[str] | None = None) -> AnnData
```

Run the whole analysis, from quality control to Leiden clusters, in one call, out of core.

Copies the counts from the file into the project folder without holding the matrix in memory, then runs quality control, the cell and gene filters, normalization, gene selection, PCA, the neighbor graph and Leiden clustering. It applies the same steps, in the same order and with the same defaults, as the step-by-step functions, so both analyze the same cells and genes. The workflow (`set_workflow`) decides the gene selection, normalization and scaling.

The filters keep every cell and gene unless you set them.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str | required | The `.h5ad` file. |
| `n_hvg` | int | `2000` | The number of genes to select. |
| `n_pcs` | int | `50` | The number of principal components. |
| `n_neighbors` | int | `15` | Neighbors per cell. |
| `target_sum` | float | `10000.0` | The total each cell is scaled to. |
| `resolution` | float | `1.0` | Leiden resolution. |
| `seed` | int | `42` | Random seed (the step functions use 0 by default). |
| `min_genes` | int | `0` | The fewest detected genes a cell may have, as in `ps.pp.filter_cells`. |
| `max_genes` | float | `inf` | The most detected genes a cell may have. |
| `min_counts` | float | `0` | The fewest total counts a cell may have. |
| `max_counts` | float | `inf` | The most total counts a cell may have. |
| `max_pct_mt` | float | `inf` | The largest share of mitochondrial counts, in percent. |
| `mito_prefix` | str | `'MT-'` | The start of mitochondrial gene names, as in `ps.pp.calculate_qc_metrics`. |
| `min_cells` | int | `0` | The fewest kept cells a gene must be detected in, as in `ps.pp.filter_genes`. Removed genes take no part in normalization, gene selection or PCA (`var['kept']`). |
| `include` | list[str] \| None | `None` | Genes to keep whatever the ranking says, added to the top `n_hvg`. A name not in the file, or a gene `min_cells` removes, is an error naming it. |

**Returns:** The analysis, with the quality metrics in `obs` and `var`, `obsm['X_pca']`, the neighbor graph and the Leiden clusters.

### `ps.set_workflow`

```python
ps.set_workflow(name: str) -> None
```

Choose the default settings of the analysis steps.

`'polariseq'` (the default): genes are ranked as Scanpy's seurat flavor ranks them, among the genes detected in more than 1% of the cells. In the one-call pipeline, each cell is normalized again over the selected genes (to 1,000, then log, as Scarf does) and each gene is standardized before PCA. UMAP starts from the multilevel spectral start, with a minimum distance of 0.5 and 100 rounds above 10,000 cells. On the human fetal atlas these defaults gave clusters closer to the published cell types (normalized mutual information 0.80 against 0.55) and one coherent UMAP of the 4 million cells.

`'scanpy'`: the steps as Scanpy defines them: the seurat ranking over all genes, no second normalization, and UMAP from the spectral start with a minimum distance of 0.1 and 500 rounds up to 10,000 cells, otherwise 200. Use it to compare with Scanpy step by step.

An argument you give explicitly always wins over the workflow.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | `'polariseq'` or `'scanpy'`. |

### `ps.workflow`

```python
ps.workflow() -> str
```

The workflow in force, set with `set_workflow`.

**Returns:** `"polariseq"` or `"scanpy"`.

### `ps.get_build_info`

```python
ps.get_build_info() -> dict[str, Any]
```

The versions of the Python package and of the compiled core it runs.

Include it when you report a problem.

**Returns:** `python_version` (the package), `rust_core_version` (the compiled core) and `matched`: whether their release numbers agree.

## The data object

What `ps.read_h5ad` returns (an `AnnData`, or a `LazyAnnData` whose matrix is read on first use;
both have the methods below). Subset with `adata[cells]` or `adata[cells, genes]` (masks, positions
or names).

| attribute | type | what it holds |
|---|---|---|
| `X` | matrix | The expression matrix: a `SparseMatrix` in memory, a `DiskMatrix` out of core (a view of the on-disk copy with its filters and normalization), or a NumPy array. |
| `obs` | dict[str, array] | Per-cell annotations: the files' columns, quality metrics, clusters, scores. Assign a NumPy array, a list or a `pandas.Categorical` of one value per cell. |
| `var` | dict[str, array] | Per-gene annotations (gene selection statistics, `included`, ...). |
| `obs_names / var_names` | list[str] | Cell barcodes and gene names. |
| `obsm` | dict[str, array] | Per-cell embeddings: `X_pca`, `X_pca_harmony`, `X_umap`. |
| `obsp` | dict[str, sparse] | Cell-by-cell graphs: `connectivities`, `distances`. |
| `uns` | dict | Everything else: each step's parameters and results (`hvg`, `pca`, `leiden`, `scrublet`, `compare_conditions`, `rank_genes_groups`, ...). |
| `n_obs / n_vars / shape` | int / tuple | Cells, genes, and both. |
| `mode` | str | `'memory'` or `'out-of-core'`: where the analysis runs (chosen by the planner unless given to `read_h5ad`). |
| `name` | str | The dataset's folder in the project (`runs/<name>/`). |
| `analysis` | str | None | The analysis's name within its dataset (`from_counts`); `None` for the first one. |

### `adata.copy`

```python
adata.copy() -> AnnData
```

An independent copy of the analysis.

Out of core the copy is a new view of the same counts on disk; the counts themselves are not copied.

**Returns:** The copy.

### `adata.from_counts`

```python
adata.from_counts(name: str, *, genes: str = 'all') -> AnnData
```

A new analysis of the same cells, starting again from their raw counts.

For trying other settings on the same data (another number of genes, normalization or resolution) and comparing the results with `tl.compare_analyses`. Out of core nothing is copied: the new analysis reads the same counts on disk. In memory the cells' counts are read again from the files the data came from.

The cells are this analysis's current cells, after quality control and any doublet removal. Their own annotations come along: the files' columns, quality metrics, doublet scores and any column you added. What this analysis derived from its matrix does not: clusters, scores, labels, embeddings and graphs.

Each analysis keeps its results in its own folder of the project (`runs/<dataset>/analyses/<name>/`), so analyses of one dataset never replace each other's files.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | The new analysis's name (letters, digits, `. _ + -`). |
| `genes` | str | `'all'` | `"all"` (default): every gene of the source files, to filter and select again; `"current"`: this analysis's genes. |

**Returns:** The new analysis; this one is unchanged.

### `adata.save`

```python
adata.save(name: str, *, overwrite: bool = False) -> Any
```

Save the analysis in the project folder, to continue it later with `ps.load`.

The counts are not copied: the saved analysis points to the files they came from, and links the results the project folder already holds (embeddings, graphs, scores). Loading rebuilds the matrix exactly as the steps left it. If a source file is moved, loading stops and says where it was expected. The analysis is saved in `<project>/states/<name>/`.

To open the analysis in other tools, export it with `write_h5ad` or `to_anndata`.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | The saved analysis's name (letters, digits, `. _ + -`). |
| `overwrite` | bool | `False` | Replace a saved analysis of that name. |

**Returns:** The folder written.

### `adata.to_anndata`

```python
adata.to_anndata() -> Any
```

Convert to an `anndata.AnnData`, for Scanpy and the tools built on it.

`X` becomes a SciPy sparse matrix (or stays a NumPy array), and `obs`, `var`, `obsm`, `obsp` and `uns` are carried over. The result of `process_diskbacked` never held the matrix in memory, so it converts without `X`, with its per-gene statistics in `var` and every per-cell result. Needs the `anndata` package.

**Returns:** An `anndata.AnnData`.

### `adata.write_h5ad`

```python
adata.write_h5ad(path: str | os.PathLike, *, counts: bool = False) -> Any
```

Write the analysis as an `.h5ad` file for other tools (Scanpy, cellxgene, R).

`X` holds the matrix as the analysis left it: the normalized (and log-transformed) values of the current cells and genes. Out of core it is written a few blocks at a time, so an atlas is written within the memory budget. Annotations, embeddings, graphs and results follow, with Scanpy's conventions.

`uns['polariseq']` records where the raw counts came from (the source files and which of their cells and genes the file holds) and the steps that made `X`; other tools ignore it. With `counts=True` the raw counts are written too, as `layers['counts']`, for someone without the source files; the file grows accordingly.

Needs no `anndata` package. The file is written under a temporary name and renamed when complete, so an interrupted write never leaves a broken file at `path`.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| os.PathLike | required | The `.h5ad` file to write (replaced if it exists). |
| `counts` | bool | `False` | Also write the raw counts, as `layers['counts']`. |

**Returns:** The path written.

### `adata.head`

```python
adata.head(n: int = 10) -> dict
```

The raw values of the first `n` cells, read without opening the whole matrix.

For an `.h5ad` file only those cells are read. 10x files cannot be read in part, so they are read whole first; such files are usually small.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | `10` | Number of cells, from the first. |

**Returns:** A dict with `data`, `indices`, `indptr`, `n_cells` and `n_genes`: the cells' raw values as compressed sparse rows.

### `adata.sample`

```python
adata.sample(n: int = 100, seed: int = 42)
```

The raw values of `n` cells drawn at random, read without opening the whole matrix.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | `100` | Number of cells. |
| `seed` | int | `42` | Random seed. |

**Returns:** A SciPy sparse (CSR) matrix of the cells' raw values.

### `adata.validate`

```python
adata.validate() -> dict[str, Any]
```

Check the file without reading its matrix.

**Returns:** A dict: the file's `path`, `format`, whether it `exists`, its `size_mb`, the shape (`n_obs`, `n_vars`), the stored values (`nnz`, `sparsity`, `encoding`), the estimated memory (`est_ram_gb`) and how long the check took (`scan_time_s`).

### `adata.close`

```python
adata.close() -> None
```

Free the memory and temporary files the matrix holds.

Afterwards the analysis keeps only the file's description; the matrix is read again when a step next needs it.

## `ps.pp`

Preprocessing, in the usual order: quality control, filters, doublets (`scrublet`, on raw counts),
normalization, gene selection, PCA, integration (`harmony_integrate`), the neighbour graph.

### `ps.pp.calculate_qc_metrics`

```python
ps.pp.calculate_qc_metrics(adata: AnnData, *, mito_prefix: str = 'MT-') -> None
```

Quality metrics for every cell and gene.

Counts each cell's total counts, its detected genes and the share of its counts from mitochondrial genes (gene names starting with `mito_prefix`), and each gene's number of cells, total and mean counts. Use them to choose the thresholds of `filter_cells` and `filter_genes`.

Writes `obs['n_counts']`, `obs['n_genes']` and `obs['pct_counts_mt']`, and `var['n_cells']`, `var['total_counts']` and `var['mean_counts']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, before normalization. |
| `mito_prefix` | str | `'MT-'` | The start of mitochondrial gene names (`"MT-"` for human, `"mt-"` for mouse). |

### `ps.pp.filter_cells`

```python
ps.pp.filter_cells(adata: AnnData, *, min_counts: float = 0, max_counts: float = inf, min_genes: int = 0, max_genes: int = inf, max_pct_mt: float = inf, inplace: bool = True) -> AnnData | None
```

Remove cells outside quality thresholds.

Keeps the cells whose total counts, detected genes and mitochondrial share lie within the limits given; a limit left out removes nothing. Needs `calculate_qc_metrics` first.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `calculate_qc_metrics`. |
| `min_counts` | float | `0` | The fewest total counts a cell may have. |
| `max_counts` | float | `inf` | The most total counts a cell may have. |
| `min_genes` | int | `0` | The fewest detected genes a cell may have. |
| `max_genes` | int | `inf` | The most detected genes a cell may have. |
| `max_pct_mt` | float | `inf` | The largest share of mitochondrial counts, in percent (needs mitochondrial genes in the data). |
| `inplace` | bool | `True` | Remove the cells from `adata` (default); `False` returns a filtered copy and leaves `adata` as it is. |

**Returns:** `None`, or the filtered copy with `inplace=False`.

### `ps.pp.filter_genes`

```python
ps.pp.filter_genes(adata: AnnData, *, min_cells: int = 0, min_counts: float = 0, inplace: bool = True) -> AnnData | None
```

Remove genes detected in too few cells.

Keeps the genes detected in at least `min_cells` cells and with at least `min_counts` counts in total, and writes the remaining genes' `var['n_cells']`, `var['total_counts']` and `var['mean_counts']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis. |
| `min_cells` | int | `0` | The fewest cells a gene must be detected in. |
| `min_counts` | float | `0` | The fewest total counts a gene must have. |
| `inplace` | bool | `True` | Remove the genes from `adata` (default); `False` returns a filtered copy and leaves `adata` as it is. |

**Returns:** `None`, or the filtered copy with `inplace=False`.

### `ps.pp.harmony_integrate`

```python
ps.pp.harmony_integrate(adata: AnnData, batch_key: str | list[str], *, use_rep: str = 'X_pca', key_added: str = 'X_pca_harmony', theta: float | list[float] = 2.0, sigma: float = 0.1, lamb: float | list[float] | None = None, n_clusters: int | None = None, tau: float = 0.0, block_size: float = 0.05, max_iter_harmony: int = 10, max_iter_kmeans: int = 4, epsilon_cluster: float = 0.001, epsilon_harmony: float = 0.01, alpha: float = 0.2, seed: int = 0) -> None
```

Correct the PCA embedding for batch effects with Harmony.

Moves the cells in the principal components so that batches mix while cell types stay apart. Cells are grouped softly into clusters that are pushed to hold every batch in proportion; within each cluster each batch's offset is estimated, and cells move back by their clusters' offsets. Grouping and correction alternate until the result stops improving.

Writes the corrected embedding to `obsm[key_added]` (`X_pca_harmony`, as in Scanpy) and the settings to `uns['harmony']`; the uncorrected embedding stays in `use_rep`, so mixing can be compared before and after. Build the graph on the corrected one.

The method is Harmony (Korsunsky et al. 2019), as R harmony 2 and harmonypy 2 run it; the defaults are harmonypy's.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pca`. |
| `batch_key` | str \| list[str] | required | The `obs` column of each cell's batch, or a list of columns to correct for together (for example `["donor", "lane"]`). |
| `use_rep` | str | `'X_pca'` | The `obsm` embedding to correct. |
| `key_added` | str | `'X_pca_harmony'` | The `obsm` key of the corrected embedding. |
| `theta` | float \| list[float] | `2.0` | How strongly to mix the batches, one value for all batch variables or one each. |
| `sigma` | float | `0.1` | Width of the soft clustering; smaller is closer to hard clusters. |
| `lamb` | float \| list[float] \| None | `None` | Ridge penalty, one for all variables or one each. `None` estimates it per cluster, as harmonypy 2 does (`alpha` times the expected number of cells). |
| `n_clusters` | int \| None | `None` | Number of clusters (default `min(n_cells / 30, 100)`). |
| `tau` | float | `0.0` | Protection of small batches against overcorrection (0: off). |
| `block_size` | float | `0.05` | The share of cells reassigned at a time. |
| `max_iter_harmony` | int | `10` | The most rounds of clustering and correction. |
| `max_iter_kmeans` | int | `4` | Assignment rounds per clustering. |
| `epsilon_cluster` | float | `0.001` | Convergence threshold of the assignment rounds. |
| `epsilon_harmony` | float | `0.01` | Convergence threshold of the rounds of clustering and correction. |
| `alpha` | float | `0.2` | Scale of the estimated ridge penalty. |
| `seed` | int | `0` | Random seed. |

### `ps.pp.highly_variable_genes`

```python
ps.pp.highly_variable_genes(adata: AnnData, n_top_genes: int = 2000, *, flavor: str | None = None, subset: bool = False, min_cells: int | None = None, n_bins: int | None = None, lowess_frac: float = 0.1, include: list[str] | None = None) -> None
```

Select the genes that vary most between cells.

Ranks the genes by how much more variable they are than genes of similar mean expression and marks the top `n_top_genes` in `var['highly_variable']`. With `subset=True` only those genes are kept, as most analyses do before PCA. Run it on log-normalized data.

Two rankings:

- `flavor='seurat'`: Seurat's dispersion method (Satija et al. 2015), computed step for step as Scanpy computes it, so the selected genes agree with Scanpy's on the same data. Writes `var['means']`, `var['dispersions']` and `var['dispersions_norm']`.
- `flavor='scarf'`: Scarf's ranking (Dhapola et al. 2022). Genes are binned by mean expression, a smooth curve (LOWESS) is fitted through each bin's least variable gene, and genes are ranked by their variance above that curve. Writes `var['log_mean']`, `var['log_var']`, `var['corrected_var']` and `var['detected']`.

In the default `'polariseq'` workflow, genes are ranked as `'seurat'` among those detected in more than 1% of the cells, so that genes seen in a handful of cells do not crowd out the informative ones. The settings are kept in `uns['hvg']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `log1p`. |
| `n_top_genes` | int | `2000` | The number of genes to select. |
| `flavor` | str \| None | `None` | `'seurat'` or `'scarf'`. Left out, the workflow decides (`ps.set_workflow`): `'polariseq'` ranks as `'seurat'` among the genes detected in more than 1% of the cells, `'scanpy'` as `'seurat'` over every gene. |
| `subset` | bool | `False` | Keep only the selected genes. |
| `min_cells` | int \| None | `None` | A gene must be detected in more cells than this to be ranked. For `'scarf'` the default is 1% of the cells; for `'seurat'` it is no floor, as in Scanpy. |
| `n_bins` | int \| None | `None` | The number of mean-expression bins (default 20 for `'seurat'`, 200 for `'scarf'`). |
| `lowess_frac` | float | `0.1` | `'scarf'` only: the share of the bins' genes in each local fit of the curve. |
| `include` | list[str] \| None | `None` | Genes to keep whatever the ranking says, by name. They are added to the top `n_top_genes` and marked in `var['included']`. A name that is not among the current genes is an error that says why and suggests the closest names. |

### `ps.pp.log1p`

```python
ps.pp.log1p(adata: AnnData | LazyAnnData) -> None
```

Take the logarithm of the expression values, `log(1 + x)`.

Compresses the range of the values, so that a few very high counts do not dominate. Zeros stay zero, so a sparse matrix stays sparse. Writes `uns['log1p']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis, after `normalize_total`. |

### `ps.pp.neighbors`

```python
ps.pp.neighbors(adata: AnnData, n_neighbors: int = 15, *, use_rep: str = 'X_pca', n_pcs: int | None = None, metric: str = 'euclidean', approximate: bool | None = None, seed: int = 42) -> None
```

Build the neighbor graph: each cell's nearest cells in an embedding.

Finds each cell's `n_neighbors` nearest cells in `obsm[use_rep]` (the PCA embedding by default, or a corrected one such as `X_pca_harmony`) and turns them into the weighted graph that clustering and UMAP work on. Above 10,000 cells the neighbors are found approximately with NN-Descent (Dong, Moses and Li 2011), which is far faster; below, exactly.

Writes `obsp['distances']` (each cell's neighbors and their distances), `obsp['connectivities']` (the weighted graph, built as UMAP builds it; McInnes et al. 2018) and `uns['neighbors']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pca`. |
| `n_neighbors` | int | `15` | Neighbors per cell. |
| `use_rep` | str | `'X_pca'` | The `obsm` embedding to search. |
| `n_pcs` | int \| None | `None` | How many of its columns to use (default: all). |
| `metric` | str | `'euclidean'` | `'euclidean'` or `'cosine'`. |
| `approximate` | bool \| None | `None` | `True` always uses NN-Descent, `False` always the exact search; left out, the exact search is used up to 10,000 cells. |
| `seed` | int | `42` | Random seed of NN-Descent. |

### `ps.pp.normalize_total`

```python
ps.pp.normalize_total(adata: AnnData, target_sum: float = 10000.0) -> None
```

Scale each cell to the same total count.

Divides each cell's counts by its total and multiplies them by `target_sum`, so that cells sequenced to different depths can be compared. Usually followed by `log1p`. Out of core the counts on disk are left unchanged: the scale is applied as they are read.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, on raw counts. |
| `target_sum` | float | `10000.0` | The total each cell is scaled to. |

### `ps.pp.pca`

```python
ps.pp.pca(adata: AnnData, n_components: int = 50, *, center: bool = True, scale: bool = False, max_value: float | None = None, seed: int = 0) -> None
```

Principal component analysis of the expression matrix.

Finds the directions along which the cells vary most and gives each cell its position along the first `n_components` of them: the embedding the neighbor graph and Harmony work on. Run it after gene selection. With `scale=True` each gene is standardized first, so that highly expressed genes do not dominate the components.

Writes `obsm['X_pca']` (cells by components) and `uns['pca']`: `variance_ratio`, the share of the variance each component explains, and `n_components`.

The components are found with randomized SVD (Halko et al. 2011) for large matrices and with an exact decomposition for smaller ones.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis. |
| `n_components` | int | `50` | Number of components. |
| `center` | bool | `True` | Subtract each gene's mean first, as PCA usually does. |
| `scale` | bool | `False` | Standardize each gene to unit variance first, as `sc.pp.scale` does; implies `center`. The matrix itself is left unchanged. |
| `max_value` | float \| None | `None` | With `scale`, the largest absolute scaled value kept; larger ones are clipped. |
| `seed` | int | `0` | Random seed of the randomized method. |

### `ps.pp.renormalize`

```python
ps.pp.renormalize(adata: AnnData, target_sum: float = 1000.0, *, log1p: bool = True, counts: AnnData | None = None) -> None
```

Normalize each cell again over the selected genes only.

After gene selection with `subset=True`, scales each cell to `target_sum` by its total over the selected genes, then takes the logarithm. This is Scarf's normalization (Dhapola et al. 2022), and the default recipe's step before `pca(scale=True)`.

In memory the values are turned back into counts and normalized again, using the totals that `normalize_total` recorded; pass `counts` only when cells were removed after normalizing. Out of core the normalization is a view on the counts on disk and is simply replaced.

Writes `uns['normalize']` and `uns['log1p']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `highly_variable_genes(..., subset=True)`. |
| `target_sum` | float | `1000.0` | The total each cell is scaled to. |
| `log1p` | bool | `True` | Take the logarithm after scaling. |
| `counts` | AnnData \| None | `None` | In memory, the raw counts of the same cells (genes matched by name), when cells were removed after `normalize_total`. |

### `ps.pp.scrublet`

```python
ps.pp.scrublet(adata: AnnData | LazyAnnData, *, batch_key: str | None = None, sim_doublet_ratio: float = 2.0, expected_doublet_rate: float = 0.05, stdev_doublet_rate: float = 0.02, n_neighbors: int | None = None, n_prin_comps: int = 30, min_counts: float = 3, min_cells: int = 3, min_gene_variability_pctl: float = 85.0, threshold: float | None = None, random_state: int = 0) -> None
```

Find droplets that held two cells (doublets) with Scrublet.

Doublets are simulated by adding up the counts of random pairs of cells, and each cell is scored by the share of simulated doublets among its nearest neighbors. Cells scoring above a threshold are called doublets; the threshold is found from the simulated doublets' scores unless you give one.

Doublets form within one capture (one 10x channel), so score each capture on its own by passing the `obs` column that names it as `batch_key`. Scrublet works on raw counts: in memory, run it before `normalize_total`; out of core it reads the counts from the copy on disk, so it can run at any point.

Writes `obs['doublet_score']`, `obs['predicted_doublet']` and `uns['scrublet']`: for each capture, the threshold, the detected and expected doublet rates, and the simulated doublets' scores, which `ps.pl.scrublet_score_distribution` draws. The names are those of Scanpy's `pp.scrublet`.

The method is Scrublet (Wolock, Lopez and Klein 2019), as its reference implementation computes it.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis, on raw counts. |
| `batch_key` | str \| None | `None` | The `obs` column of the capture each cell came from. |
| `sim_doublet_ratio` | float | `2.0` | Doublets simulated per cell. |
| `expected_doublet_rate` | float | `0.05` | The share of doublets you expect from the loading density. |
| `stdev_doublet_rate` | float | `0.02` | The uncertainty of that share. |
| `n_neighbors` | int \| None | `None` | Neighbors per cell (default `round(0.5 * sqrt(n))`). |
| `n_prin_comps` | int | `30` | Principal components used to find the neighbors. |
| `min_counts` | float | `3` | A gene is used if it reaches `min_counts` normalized counts in at least `min_cells` cells. |
| `min_cells` | int | `3` | A gene is used if it reaches `min_counts` normalized counts in at least `min_cells` cells. |
| `min_gene_variability_pctl` | float | `85.0` | Genes are used above this percentile of variability. |
| `threshold` | float \| None | `None` | The score above which a cell is called a doublet. |
| `random_state` | int | `0` | Random seed of the simulation. |

## `ps.tl`

Tools on the neighbour graph and the clusters: `leiden`, `umap`, markers (`rank_genes_groups`),
annotation and curation, `paris`, `paga`, `dpt`, and comparisons of conditions
(`compare_conditions`) and of analyses (`compare_analyses`).

### `ps.tl.annotate`

```python
ps.tl.annotate(adata: AnnData, mapping: dict, groupby: str = 'leiden', *, key_added: str = 'cell_type', unassigned: str | None = None) -> None
```

Name clusters by hand.

`mapping` maps clusters to names, `{"0": "CD4 T cell", "4": "B cell"}`, or names to lists of clusters, `{"B cell": ["4", "9"]}`. Clusters given the same name become one group, the simplest way to join clusters that are one cell type. A cluster left out of `mapping` keeps its own label, or takes `unassigned` when that is given. Writes the names to `obs[key_added]`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, clustered. |
| `mapping` | dict | required | `{cluster: name}` or `{name: [clusters]}`. |
| `groupby` | str | `'leiden'` | The `obs` column the clusters come from. |
| `key_added` | str | `'cell_type'` | The `obs` column the names are written to. |
| `unassigned` | str \| None | `None` | The name of clusters left out of `mapping` (default: their own label). |

### `ps.tl.auto_annotate`

```python
ps.tl.auto_annotate(adata: AnnData, groupby: str = 'leiden', *, source: str | list[str] | tuple[str, ...] | None = ('curated', 'hpa'), tissue: str | list[str] | None = None, species: str | None = None, markers: Any = None, max_markers: int | None = 30, margin: float = 0.05, key_added: str = 'cell_type', unassigned: str = 'unassigned', method: str | None = None) -> Any
```

Name each cluster's cell type from marker genes, with a score.

Every cell is scored for every cell type in the chosen marker databases, and each cluster takes the type its cells score highest for on average. The names are a first guess: check them against each cluster's marker genes (`ps.get.markers`, a dot plot). Give `tissue` when you know it; it removes look-alike types of other organs, such as liver macrophages for blood monocytes.

How the scores are made, following ScType (Ianevski, Giri and Aittokallio 2022) and the Cell Marker Accordion (Busarello et al. 2025):

1. Markers: the chosen sources' tables (`ps.marker_table` shows them), narrowed to `tissue`, plus your own `markers`. Ribosomal, mitochondrial and dissociation-stress genes are left out. For mouse data, human markers are mapped to mouse genes.
2. Weights: a marker counts more when many records name it and when few cell types share it. Each type keeps its `max_markers` best markers.
3. Scores: each marker is standardized over all cells, using every gene of the source files, so markers outside the selected genes still count. A cell's score for a type adds up its markers and subtracts the genes the type should not express.
4. Clusters: a cluster stays `unassigned` when its best score is not above zero or fewer than a quarter of its cells support it. When the runner-up is within `margin` of the best, the cluster takes their nearest shared Cell Ontology type ("T cell" for two kinds of T cell).
5. Cells: every cell also gets its own best type, so a small population hidden inside a larger cluster shows in the table's `also` column.

On published labels (PBMC 3k, Kang et al. 2018, the Cao et al. 2020 fetal atlas organ by organ), the default named 73% of cells in the right lineage on average, 77% with `tissue`, and 80% with the Accordion added.

Marker sources (`source=`, one or a list; types are merged by Cell Ontology term):

- `"curated"`: a short list of canonical markers for about 40 types, with the blood subtypes.
- `"hpa"`: the Human Protein Atlas single cell types, 154 types in 31 tissues.
- `("curated", "hpa")`: both, the default, shipped with Polariseq.
- `"msigdb_c8"`: MSigDB C8 cell type signatures from 22 single-cell studies.
- `"accordion"`, `"cellmarker"`, `"panglaodb"`, `"sctype"`: downloaded once on first use (their terms do not allow shipping them). `"accordion"` needs `pyreadr`, `"cellmarker"` and `"sctype"` need `openpyxl` (both: `pip install 'polariseq[markers]'`).
- `"celltypist"`: CellTypist's trained models in place of markers (needs `celltypist`).

Writes the names to `obs[key_added]`, with `_ontology_id`, `_score`, `_cell` (each cell's own best type) and `_cell_score` beside it, and the details to `uns['annotation']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, clustered. |
| `groupby` | str | `'leiden'` | The `obs` column of the clusters. |
| `source` | str \| list[str] \| tuple[str, ...] \| None | `('curated', 'hpa')` | The marker source or sources above. `None` with `markers`: only your markers. |
| `tissue` | str \| list[str] \| None | `None` | Keep this tissue's types, a name or a list: `"immune"`, `"brain"`, `"lung"`, `"liver"`, `"kidney"`, `"heart"`, `"gut"`, ... (`ps.marker_table` lists them). Immune types and types found in many tissues are kept with any tissue. |
| `species` | str \| None | `None` | `"human"` or `"mouse"`; `None` guesses from the gene names. |
| `markers` | Any | `None` | Your own markers: `{"type": ["GENE", ...]}`, `{"type": {"positive": [...], "negative": [...]}}`, a table with `cell_type` and `gene` columns, or a `.gmt`, `.csv` or `.tsv` file. |
| `max_markers` | int \| None | `30` | How many of its best markers each type keeps (`None`: all). |
| `margin` | float | `0.05` | How close, as a fraction, the runner-up must score for the shared parent type to be used (0: never). |
| `key_added` | str | `'cell_type'` | The `obs` column of the names. |
| `unassigned` | str | `'unassigned'` | The name given to clusters and cells no type fits. |
| `method` | str \| None | `None` | The earlier name of `source`. |

**Returns:** One row per cluster: `cell_type`, `ontology_id`, `score`, `runner_up` and its score, `support` (the share of the cluster's cells scoring above zero for the type), `note` and `also` (groups of cells scoring highest for another type).

### `ps.tl.compare_analyses`

```python
ps.tl.compare_analyses(analyses: dict, key: str = 'leiden', *, labels: str | None = None, seeds: int = 0, n_markers: int = 20) -> Any
```

Measure how far analyses of the same cells agree.

For analyses made with different settings (`adata.from_counts`), compares their clusters on the cells they share, matched by name, with the adjusted Rand index (Hubert and Arabie 1985) and the normalized mutual information. With `seeds`, each analysis's clustering is also rerun with other seeds, so a difference between settings can be told from the variation between runs. The analyses themselves are left as they were.

| argument | type | default | meaning |
|---|---|---|---|
| `analyses` | dict | required | `{name: analysis}`, two or more. |
| `key` | str | `'leiden'` | The `obs` column of clusters in each. |
| `labels` | str \| None | `None` | An `obs` column of reference labels (for example published cell types) to score each analysis against. |
| `seeds` | int | `0` | How many other seeds to rerun each analysis's Leiden clustering with. |
| `n_markers` | int | `20` | How many top marker genes to compare between matched clusters, when both analyses ran `rank_genes_groups` on `key`. |

**Returns:** The comparison: `ari` and `nmi` for every pair, `n_clusters`, `matches` (which cluster became which) and `crossing` for each pair, and `labels` and `noise` when asked. Printing it shows a summary.

### `ps.tl.compare_conditions`

```python
ps.tl.compare_conditions(adata: AnnData | LazyAnnData, condition_key: str, reference: str, *, groupby: str | None = None, sample_key: str | None = None, conditions: list[str] | None = None, groups: list[str] | None = None, min_cells: int = 10, key_added: str = 'compare_conditions') -> Any
```

Compare conditions within each cell type: which genes change, and by how much.

With `sample_key` (recommended), the counts are added up per replicate (for example the donor), condition and cell type, and the sums are tested as bulk RNA-seq samples, the way edgeR and limma test them: the expression filter, TMM normalization and limma-trend. The replicates are the unit of evidence, so the p-values do not shrink as the number of cells grows. A donor measured under several conditions is compared with itself. The raw counts are read wherever they are: the matrix while it is still raw, the copy on disk, or the source files.

Without `sample_key`, each condition's cells are compared with the reference's by Welch's t-test. Cells from one donor are not independent, so these p-values overstate the evidence: use them to explore, and replicates to conclude.

Writes the table, a summary per cell type and the settings to `uns[key_added]`.

The method follows edgeR and limma (Robinson, McCarthy and Smyth 2010; Ritchie et al. 2015; Robinson and Oshlack 2010; Law et al. 2014; Phipson et al. 2016).

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis, with the conditions, cell types and replicates in `obs`. |
| `condition_key` | str | required | The `obs` column of the conditions. |
| `reference` | str | required | The condition the others are compared with. |
| `groupby` | str \| None | `None` | The `obs` column of the cell types (default: all cells as one group). |
| `sample_key` | str \| None | `None` | The `obs` column of the replicates. |
| `conditions` | list[str] \| None | `None` | The conditions to compare (default: all but the reference). |
| `groups` | list[str] \| None | `None` | The cell types to compare (default: all). |
| `min_cells` | int | `10` | With `sample_key`, the fewest cells a replicate needs in a cell type to take part; without, the fewest cells each condition needs. |
| `key_added` | str | `'compare_conditions'` | Where in `uns` the results go. |

**Returns:** A `pandas.DataFrame` with one row per cell type, condition and gene, most significant first: `log2_fold_change`, the mean expression on each side, the share of cells expressing the gene on each side, `t`, `p_value` and `adj_p_value` (Benjamini-Hochberg, within each cell type and condition).

### `ps.tl.dpt`

```python
ps.tl.dpt(adata: AnnData, root: int = 0) -> None
```

Pseudotime from a root cell: each cell's shortest-path distance from it.

The paths run over the neighbor graph of `pp.neighbors`, with its distances as edge lengths and every edge usable in both directions. The distances are divided by the largest, so the root is at 0 and the farthest cell it reaches at 1; cells it cannot reach get `inf`. This is a graph distance; diffusion pseudotime (Haghverdi et al. 2016) is a different measure.

Writes `obs['dpt_pseudotime']`, and `uns['dpt']` with the root and the number of cells reached.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `root` | int | `0` | Position of the root cell. |

### `ps.tl.leiden`

```python
ps.tl.leiden(adata: AnnData, resolution: float = 1.0, *, seed: int = 0, key_added: str = 'leiden', method: str = 'auto') -> None
```

Cluster the cells with the Leiden algorithm.

Groups cells that are densely connected in the neighbor graph of `pp.neighbors` (`obsp['connectivities']`). Raise `resolution` for more, smaller clusters; lower it for fewer, larger ones. The same graph and seed always give the same clusters, at any number of threads.

Writes the cluster numbers to `obs[key_added]` and the run's details to `uns['leiden']`: the number of clusters (`n_communities`), the modularity of the clustering (`quality`), the resolution and the method.

The algorithm is Leiden (Traag et al. 2019), as the authors' CWTS implementation runs it. `method='fast'` runs the earlier Louvain method (Blondel et al. 2008) on the same graph, and `method='igraph'` runs igraph's Leiden, which needs the `python-igraph` package.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `resolution` | float | `1.0` | Higher gives more, smaller clusters; lower gives fewer, larger ones. |
| `seed` | int | `0` | Random seed; the same seed gives the same clusters. |
| `key_added` | str | `'leiden'` | The `obs` column the cluster numbers are written to. |
| `method` | str | `'auto'` | `'auto'` (Leiden), `'fast'` (Louvain) or `'igraph'` (igraph's Leiden). |

### `ps.tl.merge_clusters`

```python
ps.tl.merge_clusters(adata: AnnData, groups: list, groupby: str = 'leiden', *, key_added: str | None = None, names: list[str] | None = None) -> None
```

Join clusters into one.

Each inner list of `groups` becomes one cluster, named after its members joined by `+` (`"3+7"`) or by the matching entry of `names`; a single list is one group. Every other cluster keeps its label. Writes the result to `obs[key_added]`, by default `obs['<groupby>_curated']`, so the original clusters stay as they were.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, clustered. |
| `groups` | list | required | The clusters to join: a list of lists, or one list for a single group. |
| `groupby` | str | `'leiden'` | The `obs` column the clusters come from. |
| `key_added` | str \| None | `None` | The `obs` column written (`groupby` replaces the original). |
| `names` | list[str] \| None | `None` | A name for each joined cluster. |

### `ps.tl.paga`

```python
ps.tl.paga(adata: AnnData, groupby: str = 'leiden') -> None
```

A graph between clusters: how strongly the neighbor graph links each pair.

For two clusters, the connectivity is the number of neighbor-graph edges between them, counted in both directions, divided by the number of cells in the two; the diagonal is 1. PAGA (Wolf et al. 2019) measures it differently, comparing the edges between two clusters with the number expected by chance.

Writes `uns['paga']`: `connectivities` (clusters by clusters), `cluster_sizes` and `groupby`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `groupby` | str | `'leiden'` | The `obs` column of numbered clusters, as `tl.leiden` writes it. |

### `ps.tl.paris`

```python
ps.tl.paris(adata: AnnData, n_clusters: int | None = None, *, balanced: bool = False, min_size: int | None = None, max_size: int | None = None, max_distance_fc: float = 2.0, key_added: str = 'paris', recompute: bool = False) -> None
```

Build the tree of clusters with Paris, and cut it.

Paris (Bonald et al. 2018) builds the whole hierarchy of the neighbor graph in one pass: a tree in which every cut is a clustering, from one cluster down to single cells. Coarse types and their subtypes are then cuts of the same tree, without clustering again. The tree matches scikit-network's `Paris` exactly.

With `n_clusters`, the tree is cut straight into that many clusters (or a few more where heights tie), numbered from 0 by decreasing size. With `balanced=True`, it is cut as Scarf does (Dhapola et al. 2022), into clusters of at most `max_size` cells.

Writes the tree to `uns[key_added]['dendrogram']` (in the layout of SciPy's `linkage`), reused while the graph is unchanged, and the labels of a cut to `obs[key_added]`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `n_clusters` | int \| None | `None` | Cut straight into this many clusters. |
| `balanced` | bool | `False` | Use Scarf's balanced cut. |
| `min_size` | int \| None | `None` | Balanced cut: the size above which two halves' heights are compared. |
| `max_size` | int \| None | `None` | Balanced cut: the largest cluster. |
| `max_distance_fc` | float | `2.0` | Balanced cut: the largest ratio of heights at which two halves are joined. |
| `key_added` | str | `'paris'` | The `obs` and `uns` key. |
| `recompute` | bool | `False` | Rebuild the tree even when one is stored. |

### `ps.tl.rank_genes_groups`

```python
ps.tl.rank_genes_groups(adata: AnnData, groupby: str = 'leiden', *, groups: list[str] | None = None, reference: str = 'rest', n_genes: int = 50) -> None
```

Find the marker genes of each group.

Compares each group's cells with all other cells (`reference="rest"`) or with one group, gene by gene, with Welch's t-test (Welch 1947) as Scanpy's `rank_genes_groups(method="t-test")` does, and ranks the genes by their t-statistic. P-values are adjusted for the number of genes tested (Benjamini and Hochberg 1995).

Writes `uns['rank_genes_groups']` (for each group, the top genes with their scores, log fold changes and p-values) and `uns['rank_genes_groups_params']`. `ps.get.markers` turns it into a table. To compare conditions with donors as replicates, use `compare_conditions`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, log-normalized. |
| `groupby` | str | `'leiden'` | The `obs` column of the groups: clusters, or any labels. |
| `groups` | list[str] \| None | `None` | The groups to rank (default: all but the reference). |
| `reference` | str | `'rest'` | `"rest"`, or the group the others are compared with. |
| `n_genes` | int | `50` | How many top genes to keep per group. |

### `ps.tl.score_genes`

```python
ps.tl.score_genes(adata: AnnData, gene_list: list[str], score_name: str | None = None, *, seed: int = 42) -> None
```

Score each cell for a gene set.

A cell's score is its mean expression of the genes in `gene_list` minus its mean expression of control genes: `max(50, 5 × the set's size)` genes drawn at random from all the other genes. Seurat's module score (Tirosh et al. 2016) works the same way but draws its control genes from genes of similar mean expression. Genes of the set that are not in the data are left out, with a warning. Runs in memory only.

Writes the scores to `obs[score_name]`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, log-normalized, in memory. |
| `gene_list` | list[str] | required | The genes of the set. |
| `score_name` | str \| None | `None` | The `obs` column of the scores (default: `score_` and the first gene's name). |
| `seed` | int | `42` | Random seed of the control genes. |

### `ps.tl.subcluster`

```python
ps.tl.subcluster(adata: AnnData, cluster: Any, groupby: str = 'leiden', *, resolution: float = 1.0, seed: int = 0, key_added: str | None = None) -> int
```

Split one cluster into smaller ones.

Runs Leiden on the neighbor graph between the cluster's cells only. Cluster `"5"` becomes `"5.0"`, `"5.1"` and so on, largest first; every other cell keeps its cluster. Writes the result to `obs[key_added]`, by default `obs['<groupby>_curated']`, so the original clusters stay as they were. Only the graph is read, so it works the same out of core.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors` and clustering. |
| `cluster` | Any | required | The cluster to split. |
| `groupby` | str | `'leiden'` | The `obs` column it belongs to. |
| `resolution` | float | `1.0` | Leiden resolution within the cluster (higher: more parts). |
| `seed` | int | `0` | Random seed. |
| `key_added` | str \| None | `None` | The `obs` column written (`groupby` replaces the original). |

**Returns:** The number of parts.

### `ps.tl.umap`

```python
ps.tl.umap(adata: AnnData, *, min_dist: float | None = None, spread: float = 1.0, n_epochs: int = 0, seed: int = 42, deterministic: bool = False, init: str | np.ndarray | None = None, init_smooth: int = 30) -> None
```

Lay the cells out in two dimensions with UMAP.

Places the cells so that neighbors in the graph of `pp.neighbors` stay close (McInnes, Healy and Melville 2018). The layout runs on all threads and differs slightly from run to run, as umap-learn's does; `deterministic=True` runs it on one thread for the same layout every time.

By default the layout starts from Polariseq's multilevel spectral start: similar cells are grouped, the standard spectral start is computed for the much smaller graph of groups, and each cell starts at its group's position, smoothed over the full graph. This is fast for millions of cells and keeps the large-scale structure.

Writes `obsm['X_umap']` and `uns['umap']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `min_dist` | float \| None | `None` | How tightly points may pack (default 0.5 in the `'polariseq'` workflow, 0.1 in `'scanpy'`). |
| `spread` | float | `1.0` | The scale of the layout. |
| `n_epochs` | int | `0` | Optimization rounds (0: 500 up to 10,000 cells, otherwise 100 in the `'polariseq'` workflow and 200 in `'scanpy'`). |
| `seed` | int | `42` | Random seed. |
| `deterministic` | bool | `False` | Run on one thread, giving the same layout every time. |
| `init` | str \| np.ndarray \| None | `None` | Where the cells start: `'multilevel'` (the default in the `'polariseq'` workflow), `'spectral'` (the exact spectral start, the default in `'scanpy'`), `'leiden'` (the multilevel start through the Leiden clusters), `'kmeans'` (Scarf's start), `'pca'` (the first two principal components), `'random'`, or an array with two columns. |
| `init_smooth` | int | `30` | Smoothing steps of the multilevel starts. |

## `ps.get`

Tables taken out of an analysis.

### `ps.get.markers`

```python
ps.get.markers(adata: AnnData, n: int = 10, *, groups: list | None = None, wide: bool = False) -> Any
```

The top marker genes of each cluster, as a table.

Reads what `tl.rank_genes_groups` found. The long table has one row per cluster and gene; `wide=True` gives one column of gene names per cluster, the table to read when naming clusters.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `tl.rank_genes_groups`. |
| `n` | int | `10` | Genes per cluster. |
| `groups` | list \| None | `None` | The clusters to include (default: all that were ranked). |
| `wide` | bool | `False` | One column of gene names per cluster. |

**Returns:** A `pandas.DataFrame`: `group`, `rank`, `gene`, `score`, `logfoldchange`, `pval_adj` and `pct` (the share of the cluster's cells expressing the gene); with `wide=True`, one column of gene names per cluster.

### `ps.get.pseudobulk`

```python
ps.get.pseudobulk(adata: AnnData | LazyAnnData, sample_key: str, *, groupby: str | None = None, condition_key: str | None = None, min_cells: int = 0) -> AnnData
```

Raw counts added up per replicate, and per cell type and condition.

The table edgeR, DESeq2 or limma test: one row per combination of `groupby`, `condition_key` and `sample_key` with at least `min_cells` cells, the gene counts summed, and the number of cells in `obs['n_cells']`. The raw counts are read wherever they are held, as in `tl.compare_conditions`. Pass the result's `to_anndata()` to pydeseq2, or write its `X` for R.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis. |
| `sample_key` | str | required | The `obs` column of the replicates (for example the donor). |
| `groupby` | str \| None | `None` | The `obs` column of the cell types (default: all cells as one group). |
| `condition_key` | str \| None | `None` | The `obs` column of the conditions, to keep them apart. |
| `min_cells` | int | `0` | Combinations with fewer cells are left out. |

**Returns:** An analysis with one row per combination (`obs` names its cell type, condition and replicate, and holds `n_cells`), the summed counts in `X` and the genes in `var_names`.

## `ps.pl`

Plots. Every function takes `show=` and `save=` (a file path); most take `ax=` to draw into an
existing Matplotlib axis.

### `ps.pl.categorical_palette`

```python
ps.pl.categorical_palette(n: int, washed: bool = True) -> list[str]
```

Colors for `n` groups, as the plots use them.

Eight base colors, then the same hues lighter and then darker, for up to 24 groups; beyond that, hues spaced evenly around the color wheel. The `classic` theme uses Paul Tol's colors.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | required | Number of colors. |
| `washed` | bool | `True` | Lighten the colors, as the theme does; `False` gives the full colors. |

**Returns:** A list of hex colors.

### `ps.pl.cluster_sizes`

```python
ps.pl.cluster_sizes(adata: AnnData, key: str = 'leiden', *, log: bool | None = None, show: bool = True, save: str | None = None, figsize: tuple | None = None, dpi: int | None = None, **kw) -> Any
```

Cells per cluster, largest first. The axis turns logarithmic when sizes span more than fifty-fold, and cluster names are written out only while they fit (at most 40); labels are plain numbers at every scale.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `key` | str | `'leiden'` | `obs` column of the clusters. |
| `log` | bool \| None | `None` | Logarithmic count axis; `None` chooses it when sizes span widely. |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `figsize` | tuple \| None | `None` | Figure size in inches, `(width, height)`. |
| `dpi` | int \| None | `None` | Figure resolution. |

**Returns:** The Matplotlib axis.

### `ps.pl.compare_analyses`

```python
ps.pl.compare_analyses(analyses: dict, key: str = 'leiden', *, basis: str = 'X_umap', ncols: int | None = None, figsize: tuple | None = None, dpi: int | None = None, show: bool = True, save: str | None = None, **kw: Any) -> Any
```

Each analysis's clusters side by side, on one embedding.

The cells are placed where the first analysis holding `basis` places them (matched by name), so the panels differ only by how each analysis grouped the same cells; each title gives the analysis's clusters and its adjusted Rand index with the first analysis.

| argument | type | default | meaning |
|---|---|---|---|
| `analyses` | dict | required | `{name: analysis}`, as for `ps.tl.compare_analyses`. |
| `key` | str | `'leiden'` | The `obs` column of clusters in each. |
| `basis` | str | `'X_umap'` | The embedding the cells are placed by. |
| `ncols` | int \| None | `None` | Panels per row. |
| `figsize` | tuple \| None | `None` | Figure size in inches, `(width, height)`. |
| `dpi` | int \| None | `None` | Figure resolution. |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** A list of axes, one per analysis.

### `ps.pl.dotplot`

```python
ps.pl.dotplot(adata: AnnData, markers: list[str] | None = None, groupby: str = 'leiden', ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Genes by group as dots: the size is the share of cells expressing the gene, the color its mean expression.

Give the genes as `markers=` or, as in Scanpy, `var_names=`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `markers` | list[str] \| None | `None` | Genes to show (or `var_names=`), one column each. |
| `groupby` | str | `'leiden'` | `obs` column of the groups, one row each. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** The Matplotlib axis.

### `ps.pl.embedding`

```python
ps.pl.embedding(adata: AnnData, basis: str = 'X_umap', color: str | list[str] | None = None, **kw) -> Any
```

Draw the cells on any two-dimensional embedding in `obsm`.

`umap` and `pca` draw through it; use it directly for an embedding they do not cover (`X_tsne`, a custom projection). The leading `X_` is optional: `basis="umap"` and `basis="X_umap"` both work.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `basis` | str | `'X_umap'` | The `obsm` embedding (`'X_umap'`, `'X_pca'`, ...). |
| `color` | str \| list[str] \| None | `None` | An `obs` column or a gene; a list draws one panel each. |

**Returns:** The Matplotlib axis; with several `color` values or `split_by`, a list of axes, one per panel.

### `ps.pl.fold_changes`

```python
ps.pl.fold_changes(result: Any, genes: list[str] | None = None, *, condition: str | None = None, n_genes: int = 6, alpha: float = 0.05, ax: Any | None = None, show: bool = True, save: str | None = None, key: str = 'compare_conditions', **kw) -> Any
```

Which cell types respond: log2 fold changes, genes against cell types.

`genes` defaults to the `n_genes` most significant of each cell type. A dot marks `adj_p_value < alpha`; a hatched cell, a gene not tested there (too little expression in that cell type).

| argument | type | default | meaning |
|---|---|---|---|
| `result` | Any | required | The table `ps.tl.compare_conditions` returns, or the analysis it was run on. |
| `genes` | list[str] \| None | `None` | Genes to show (default: the most significant of each cell type). |
| `condition` | str \| None | `None` | The condition to show, when several were compared. |
| `n_genes` | int | `6` | Genes taken from each cell type when `genes` is not given. |
| `alpha` | float | `0.05` | Adjusted p-value below which a change is marked. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `key` | str | `'compare_conditions'` | Where in `uns` the comparison is, when `result` is the analysis. |

**Returns:** The Matplotlib axis.

### `ps.pl.get_theme`

```python
ps.pl.get_theme() -> dict
```

The active theme's settings: colors, font sizes, resolution, as `set_theme` sets them.

**Returns:** A copy of the settings; changing it leaves the theme as it is (`set_theme` changes it).

### `ps.pl.heatmap`

```python
ps.pl.heatmap(adata: AnnData, markers: list[str] | None = None, groupby: str = 'leiden', ax: Any | None = None, cmap: Any = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Genes by group as a heatmap, each gene's mean expression standardized across the groups.

Give the genes as `markers=` or, as in Scanpy, `var_names=`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `markers` | list[str] \| None | `None` | Genes to show (or `var_names=`). |
| `groupby` | str | `'leiden'` | `obs` column of the groups. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `cmap` | Any | `None` | Matplotlib color map (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** The Matplotlib axis.

### `ps.pl.highly_variable_genes`

```python
ps.pl.highly_variable_genes(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Each gene's mean expression against its variability, with the selected genes highlighted.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** The Matplotlib axis.

### `ps.pl.overview`

```python
ps.pl.overview(adata: AnnData, out: str, *, cluster_key: str = 'leiden', color_by: list[str] | tuple = (), thresholds: dict | None = None, qc_before: dict | None = None, prefix: str = '') -> dict
```

The standard figures of an analysis, written to `out`, each drawn the way that suits the size of the data: quality-control distributions over every cell, variance explained by the principal components, cluster sizes, and the UMAP colored by cluster and by any obs column in `color_by` (drawn as density above :data:`DENSITY_ABOVE` cells). `qc_before` is the per-cell metrics from before filtering (see `qc_distributions`), so the cells a threshold removed are shown. Also writes `figure_data.json`, the numbers behind the quality-control, cluster-size and variance plots (`overview_data`). Returns `{figure: path}` for the figures that could be drawn.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `out` | str | required | Folder the figures are written to (created if missing). |
| `cluster_key` | str | `'leiden'` | `obs` column of the clusters. |
| `color_by` | list[str] \| tuple | `()` | Further `obs` columns to color the UMAP by, one figure each. |
| `thresholds` | dict \| None | `None` | Filter thresholds to mark, `{'n_genes': 200}`. |
| `qc_before` | dict \| None | `None` | Per-cell metrics from before filtering, to show what was removed. |
| `prefix` | str | `''` | Text put before each file name. |

**Returns:** `{figure: path}` for each figure written.

### `ps.pl.overview_data`

```python
ps.pl.overview_data(adata: AnnData, *, cluster_key: str = 'leiden', qc_before: dict | None = None, thresholds: dict | None = None, bins: int = 64) -> dict
```

The numbers behind the figures of `overview`, as plain Python values.

`qc`: for each metric, the bin edges (evenly spaced on a log scale, as drawn), the number of cells in each bin before filtering (when `qc_before` is given) and among the cells kept, the threshold and how many cells fall below it; `cluster_sizes`; `pca_variance_ratio`; and the number of cells in the embedding. Written next to the figures as `figure_data.json`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `cluster_key` | str | `'leiden'` | `obs` column of the clusters. |
| `qc_before` | dict \| None | `None` | Per-cell metrics from before filtering. |
| `thresholds` | dict \| None | `None` | Filter thresholds, `{'n_genes': 200}`. |
| `bins` | int | `64` | Histogram bins (log-spaced). |

**Returns:** A dict with `qc`, `cluster_sizes`, `pca_variance_ratio` and the number of cells in the embedding, as described above.

### `ps.pl.pca`

```python
ps.pl.pca(adata: AnnData, color: str | None = None, ax: Any | None = None, size: float | None = None, title: str | None = None, cmap: str | None = None, show: bool = True, save: str | None = None, legend_loc: str | None = 'right', **kw) -> Any
```

Draw the cells on the first two principal components. Takes the same options as `umap`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `color` | str \| None | `None` | An `obs` column or a gene to color the cells by. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `size` | float \| None | `None` | Point size (default: chosen from the number of cells). |
| `title` | str \| None | `None` | Title above the plot (default: what is shown). |
| `cmap` | str \| None | `None` | Matplotlib colour map for values (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `legend_loc` | str \| None | `'right'` | `'right'`, `'on data'` (labels at the groups) or `None`. |

**Returns:** The Matplotlib axis.

### `ps.pl.pca_variance_ratio`

```python
ps.pl.pca_variance_ratio(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, n_comps: int = 30, **kw) -> Any
```

The share of the variance each principal component explains, with the running total.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `n_comps` | int | `30` | Principal components shown. |

**Returns:** The Matplotlib axis.

### `ps.pl.qc_distributions`

```python
ps.pl.qc_distributions(adata: AnnData, keys: tuple | list = ('n_genes', 'n_counts'), *, thresholds: dict | None = None, before: dict | None = None, bins: int = 64, show: bool = True, save: str | None = None, figsize: tuple | None = None, dpi: int | None = None, **kw) -> Any
```

Distributions of per-cell quality metrics, from every cell.

Counts and detected genes span orders of magnitude, so the bins are spaced evenly on a log scale (a linear histogram on a log axis collapses the bulk of the cells into a few wide blocks). The median is marked, and so is any filter threshold passed in `thresholds` (`{"n_genes": 200}`).

`adata` holds the cells that were kept. To show the ones a threshold removed, pass their metrics from before filtering in `before` (`{"n_genes": values, "n_counts": values}`, one value per cell, read right after `calculate_qc_metrics`): they are drawn in grey under the kept cells, and the subtitle counts what was removed. Without it a threshold line sits at the edge of the data and cuts nothing.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `keys` | tuple \| list | `('n_genes', 'n_counts')` | `obs` columns of per-cell metrics. |
| `thresholds` | dict \| None | `None` | Filter thresholds to mark, `{'n_genes': 200}`. |
| `before` | dict \| None | `None` | The same metrics from before filtering, `{key: values}`. |
| `bins` | int | `64` | Histogram bins (log-spaced). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `figsize` | tuple \| None | `None` | Figure size in inches, `(width, height)`. |
| `dpi` | int \| None | `None` | Figure resolution. |

**Returns:** A list of axes, one per metric in `keys`.

### `ps.pl.rank_genes_groups`

```python
ps.pl.rank_genes_groups(adata: AnnData, groups: list[str] | None = None, n_genes: int = 10, ax: Any | None = None, show: bool = True, save: str | None = None, *, ncols: int | None = None, figsize: tuple | None = None, dpi: int | None = None, sharey: bool = False, key: str = 'rank_genes_groups', **kw) -> Any
```

Top marker genes per cluster, ranked, one panel per group.

Reads what `ps.tl.rank_genes_groups` wrote. The y axis is the t-statistic, so panel heights are comparable within a group but not necessarily across them. `sharey=True` puts every panel on one scale, which flattens the clusters with weaker markers.

:

```python
ps.tl.rank_genes_groups(adata, groupby="leiden")
ps.pl.rank_genes_groups(adata, n_genes=8)
```

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `groups` | list[str] \| None | `None` | Clusters to show (default: all). |
| `n_genes` | int | `10` | Genes per cluster. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `ncols` | int \| None | `None` | Panels per row. |
| `figsize` | tuple \| None | `None` | Figure size in inches, `(width, height)`. |
| `dpi` | int \| None | `None` | Figure resolution. |
| `sharey` | bool | `False` | One y scale for every panel. |
| `key` | str | `'rank_genes_groups'` | Where in `uns` the ranking is. |

**Returns:** A list of axes, one per group; the axis given as `ax`, when one is.

### `ps.pl.scatter_qc`

```python
ps.pl.scatter_qc(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Detected genes against total counts for every cell, drawn as density for large datasets.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** The Matplotlib axis.

### `ps.pl.scrublet_score_distribution`

```python
ps.pl.scrublet_score_distribution(adata: AnnData, *, log: bool = True, show: bool = True, save: str | None = None, **kw) -> Any
```

Doublet scores of the cells and of the simulated doublets, per batch.

The threshold `ps.pp.scrublet` chose (dashed) should sit in the valley between the simulated doublets' two modes; if it does not, pass one with `ps.pp.scrublet(..., threshold=...)`. Cells are drawn on a log scale, where the few doublets would otherwise vanish beside the singlets.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `log` | bool | `True` | Logarithmic count axis for the cells. |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** A NumPy array of axes, one row per batch: the cells' scores, then the simulated doublets'.

### `ps.pl.set_theme`

```python
ps.pl.set_theme(name: str = 'paper', **overrides: Any) -> None
```

Set the look of every later figure.

Themes: `paper` (the default, journal figure sizes), `notebook` (larger text for screens), `talk` (slides), `dark` (dark backgrounds), `print` (black text, heavier lines, 600 dpi) and `classic` (the earlier default look, so older figures draw the same). Larger text comes with larger figures, so labels do not collide.

Any theme key may be overridden:

```python
ps.pl.set_theme("dark", point_scale=2.0, dpi=200)
```

Themes reset from the `paper` baseline each time, so switching between them does not accumulate the previous one's overrides.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | `'paper'` | `'paper'`, `'notebook'`, `'talk'`, `'dark'`, `'print'` or `'classic'`. |

### `ps.pl.stacked_bar`

```python
ps.pl.stacked_bar(adata: AnnData, groupby: str = 'leiden', color_by: str = 'dataset', ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

What each group is made of, as horizontal bars that add up to 100%.

For example, the share of each dataset or condition within each cluster.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `groupby` | str | `'leiden'` | `obs` column of the bars (for example the clusters). |
| `color_by` | str | `'dataset'` | `obs` column split within each bar (for example the dataset or condition). |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

**Returns:** The Matplotlib axis.

### `ps.pl.umap`

```python
ps.pl.umap(adata: AnnData, color: str | None = None, ax: Any | None = None, size: float | None = None, title: str | None = None, cmap: str | None = None, show: bool = True, save: str | None = None, legend_loc: str | None = 'right', **kw) -> Any
```

Draw the cells on their UMAP layout, colored by a column or a gene.

`color` accepts an obs column, a gene name, or a **list of either**, in which case a grid of panels is drawn (`ncols=` to control the shape):

```python
ps.pl.umap(adata, color=["leiden", "CD3E", "LYZ", "MS4A1"])
```

Extra attributes: `subtitle, caption, legend_title, palette, alpha, frame, colorbar, on_data (print category names on clusters), groups (highlight a subset, grey the rest), ncols, figsize, dpi`.

Labelling groups (`labels=`), chosen per group:

```python
ps.pl.umap(adata, color="cell_type", labels="number")     # numbered discs, names in the legend
ps.pl.umap(adata, color="cell_type", labels="name")       # names on the groups
ps.pl.umap(adata, color="cell_type", labels="callout")    # names outside, joined by lines
ps.pl.umap(adata, color="cell_type", labels="auto",       # numbers, callouts for small groups
           label_style={"Platelet": "callout", "B cell": "name"},  # per-group choice
           label_groups=["B cell", "CD4 T cell", "Platelet"])      # only these labelled
```

`callout_below` sets the size (share of cells) under which `"auto"` uses a callout. Auto-annotation workflow:

```python
ps.tl.auto_annotate(adata)                # writes obs['cell_type']
ps.pl.umap(adata, color='cell_type', on_data=True)
```

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `color` | str \| None | `None` | An `obs` column or a gene to color the cells by (a list draws one panel each); `None` draws the cells in grey. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `size` | float \| None | `None` | Point size (default: chosen from the number of cells). |
| `title` | str \| None | `None` | Title above the plot (default: what is shown). |
| `cmap` | str \| None | `None` | Matplotlib colour map for values (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `legend_loc` | str \| None | `'right'` | `'right'`, `'on data'` (labels at the groups) or `None`. |

**Returns:** The Matplotlib axis; with several `color` values or `split_by`, a list of axes, one per panel.

### `ps.pl.violin`

```python
ps.pl.violin(adata: AnnData, keys: list[str], groupby: str | None = None, ax: Any | None = None, show: bool = True, save: str | None = None, *, split_by: str | None = None, **kw) -> Any
```

Violin plots of genes or `obs` columns, per group, with each group's median marked.

With `split_by` (for example `"condition"`) each group's violin is split by that column: two conditions as the left and right halves of one violin, more as violins side by side, so a gene's change between conditions reads within each cell type.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `keys` | list[str] | required | `obs` columns or genes, one violin set each. |
| `groupby` | str \| None | `None` | `obs` column of the groups (default: all cells as one). |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `split_by` | str \| None | `None` | `obs` column of two levels (for example the condition), drawn as the two halves of each violin. |

**Returns:** The Matplotlib axis of the first panel; with `split_by`, an array of axes, one per key.

### `ps.pl.volcano`

```python
ps.pl.volcano(result: Any, group: str | None = None, *, condition: str | None = None, n_labels: int = 10, alpha: float = 0.05, min_log2_fold_change: float = 1.0, ax: Any | None = None, show: bool = True, save: str | None = None, key: str = 'compare_conditions', **kw) -> Any
```

Fold change against significance for one cell type's comparison.

`result` is the table `ps.tl.compare_conditions` returns, or the data it was run on. Genes past both thresholds (`adj_p_value < alpha` and `|log2 fold change| >= min_log2_fold_change`) are colored, up and down, and the most significant are named.

| argument | type | default | meaning |
|---|---|---|---|
| `result` | Any | required | The table `ps.tl.compare_conditions` returns, or the analysis it was run on. |
| `group` | str \| None | `None` | The cell type to show (required when the table holds several). |
| `condition` | str \| None | `None` | The condition to show, when several were compared. |
| `n_labels` | int | `10` | Genes named, the most significant first. |
| `alpha` | float | `0.05` | Adjusted p-value threshold. |
| `min_log2_fold_change` | float | `1.0` | Absolute log2 fold change threshold. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `key` | str | `'compare_conditions'` | Where in `uns` the comparison is, when `result` is the analysis. |

**Returns:** The Matplotlib axis.

## Results and helpers

### `ps.Project.clean`

```python
ps.Project.clean(part: str = 'scratch') -> int
```

Delete `scratch`, or the `latents`, `results` or `matrix` of every dataset. Returns the bytes freed. Run it only when no analysis of this project is running.

| argument | type | default | meaning |
|---|---|---|---|
| `part` | str | `'scratch'` | `'scratch'`, `'latents'`, `'results'` (named analyses' results too) or `'matrix'`. |

**Returns:** The bytes freed.

### `ps.Project.results_dir`

```python
ps.Project.results_dir(name: str, analysis: str | None = None) -> Path
```

Where the steps of an analysis of the dataset `name` keep their results: the dataset's `results/`, or for a named analysis (`from_counts`) `analyses/<analysis>/results/`.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | The dataset's folder name (`adata.name`). |
| `analysis` | str \| None | `None` | A named analysis (`adata.analysis`), or `None`. |

### `ps.Project.run_dir`

```python
ps.Project.run_dir(name: str) -> Path
```

The folder of the dataset analyzed under `name`.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | The dataset's folder name (`adata.name`). |

### `ps.Project.status`

```python
ps.Project.status() -> dict[str, int]
```

Bytes used by each part of the project, per dataset.

### `ps.Budget.summary`

```python
ps.Budget.summary() -> dict[str, Any]
```

The budget as a plain dict.

### `ps.Budget.with_overrides`

```python
ps.Budget.with_overrides(*, ram_gb: float | None = None, threads: int | None = None, spill_dir: str | None = None, spill_gb: float | None = None, allow_lossy: bool | None = None) -> Budget
```

A copy with the given fields replaced; the rest stays detected.

| argument | type | default | meaning |
|---|---|---|---|
| `ram_gb` | float \| None | `None` | Memory ceiling, in GB. |
| `threads` | int \| None | `None` | Worker threads. |
| `spill_dir` | str \| None | `None` | Folder for out-of-core scratch files. |
| `spill_gb` | float \| None | `None` | Ceiling for those files, in GB. |
| `allow_lossy` | bool \| None | `None` | Let the planner trade accuracy for fit (off by default). |

**Returns:** A new budget; the fields left as `None` keep their values.

### `Plan.report`

```python
Plan.report() -> str
```

A readable table — what `ps.plan()` prints.

### `Plan.summary`

```python
Plan.summary() -> dict[str, Any]
```

The plan as a plain dict: the shape, mode, block size, peaks, budget and every step.

### `AnalysisComparison`

What `ps.tl.compare_analyses` returns.

| attribute | meaning |
|---|---|
| `key` | The `obs` column of clusters compared. |

### `polariseq.markers.get_all_markers`

```python
polariseq.markers.get_all_markers() -> dict[str, list[str]]
```

The marker genes of every cell type in the curated table, over all tissues.

**Returns:** Dict mapping cell type name → list of marker gene symbols, merged over the tissues.

### `polariseq.markers.get_tissue_markers`

```python
polariseq.markers.get_tissue_markers(tissue: str) -> dict[str, list[str]]
```

The marker genes of one tissue's cell types, from the curated table.

| argument | type | default | meaning |
|---|---|---|---|
| `tissue` | str | required | Tissue name (e.g., 'immune', 'heart', 'brain'). Aliases like 'pbmc', 'blood', 'cardiac' are accepted. |

**Returns:** Dict mapping cell type name → list of marker gene symbols.

### `polariseq.markers.score_cell_type`

```python
polariseq.markers.score_cell_type(gene_scores: dict[str, float], markers: list[str]) -> float
```

How well a cell type's markers match a set of per-gene scores: their mean.

| argument | type | default | meaning |
|---|---|---|---|
| `gene_scores` | dict[str, float] | required | Dict mapping gene name → score (e.g., mean expression). |
| `markers` | list[str] | required | List of marker gene names for this cell type. |

**Returns:** Mean score across markers that are present in gene_scores.

## Errors

| error | raised when |
|---|---|
| `KeyError` | a column, gene, group, embedding or result named is not there (the message names the closest match and what exists) |
| `FileNotFoundError` | an input file or folder is missing (the message says where it looked) |
| `ps.SourceMovedError` | a saved analysis's source file has moved (`relink=` gives its new place) |
| `ValueError` | a value cannot be used (the message gives the limit or what is expected) |
| `ps.ProjectNotSet` | a step must write to disk and no project folder is declared |
| `ps.BudgetWarning` | a step is predicted to need more memory than the budget (a warning) |
