# Polariseq API reference

Every public function, with its arguments, their types and defaults, and what
it returns. Written from the code (`python -m polariseq._api_reference`), so it
matches the installed version: 0.2.1.

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
- **Errors name the fix**: a misspelled column, gene or file names the closest
  match; a step run too early names the step to run first.

## `ps`: project, budget, planning, reading

Declare the project folder and the budget first; read data with `read_h5ad` (or the 10x readers);
`plan` predicts the cost before anything is read; `load` reopens a saved analysis.

### `ps.project`

```python
ps.project(path: str | Path) -> Project
```

Declare the project folder of this session, creating it if needed.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| Path | required | The folder (created if missing); `~` is expanded. |

**Returns:** The `Project`, now the session's project folder.

### `ps.set_budget`

```python
ps.set_budget(*, ram_gb: float | None = None, threads: int | None = None, spill_dir: str | None = None, spill_gb: float | None = None, allow_lossy: bool | None = None) -> Budget
```

Override part of the budget; anything omitted stays as detected.

| argument | type | default | meaning |
|---|---|---|---|
| `ram_gb` | float \| None | `None` | Ceiling for in-memory data structures. |
| `threads` | int \| None | `None` | Worker threads for parallel kernels. |
| `spill_dir` | str \| None | `None` | Directory for out-of-core scratch files. |
| `spill_gb` | float \| None | `None` | Ceiling for those files. |
| `allow_lossy` | bool \| None | `None` | Permit the planner to trade accuracy for fit. **Off by default.** Polariseq will pick a slower exact plan over a faster approximate one unless asked, and reports any approximation it does make. |

**Returns:** The new active budget.

### `ps.budget`

```python
ps.budget() -> Budget
```

The active `polariseq.resources.Budget`, detecting on first use.

### `ps.resources`

```python
ps.resources() -> Resources
```

What this machine has: RAM (total and available), cores, spill space.

### `ps.plan`

```python
ps.plan(path: str | list[str] | dict[str, str], *, genes: str = 'intersection', n_hvg: int = 2000, n_pcs: int = 50, n_neighbors: int = 15, storage: str | None = None, with_umap: bool = True, verbose: bool = True)
```

Predict what analysing `path` will cost, before running anything.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| list[str] \| dict[str, str] | required | Dataset to plan for; or several, as a list of paths or a dict from each dataset's label to its path, planned as the one dataset `read_h5ad` reads from them. |
| `genes` | str | `'intersection'` | For several datasets, the genes kept: `"intersection"` (default) or `"union"` (see `read_h5ad`). |
| `n_hvg` | int | `2000` | Pipeline settings that affect the cost. |
| `n_pcs` | int | `50` | Pipeline settings that affect the cost. |
| `n_neighbors` | int | `15` | Pipeline settings that affect the cost. |
| `storage` | str \| None | `None` | Force `"ram"`, `"compressed"` or `"spill"`. |
| `with_umap` | bool | `True` | Include the UMAP step. |
| `verbose` | bool | `True` | Print the report as well as returning the plan. |

**Returns:** A `polariseq.planning.Plan` — per-step peaks, the limiting step, and whether it fits; for several datasets, each one's shape and the genes it keeps in `datasets`.

### `ps.max_cells`

```python
ps.max_cells(*, n_genes: int = 30000, nnz_per_cell: int = 2000, storage: str | None = None, **kwargs: Any) -> int
```

Largest cell count that fits the active budget, at this density.

| argument | type | default | meaning |
|---|---|---|---|
| `n_genes` | int | `30000` | Genes in the dataset. |
| `nnz_per_cell` | int | `2000` | Stored (non-zero) values per cell, the density of the data. |
| `storage` | str \| None | `None` | `None` (the planner chooses), `'ram'`, `'compressed'` or `'spill'` (out of core). |

**Returns:** The largest number of cells the active budget can analyze at that density.

### `ps.read_h5ad`

```python
ps.read_h5ad(path: str | list[str] | dict[str, str], *, sparse: bool = True, lazy: bool = True, mode: str = 'auto', name: str | None = None, batch_key: str = 'dataset', genes: str = 'intersection', obs_columns: list[str] | None = None) -> AnnData | LazyAnnData
```

Read a `.h5ad` file, or several as one dataset. With `lazy=True` (default), returns instantly with metadata; the data matrix materializes on first compute operation.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| list[str] \| dict[str, str] | required | Path to the `.h5ad` file; or several, as a list of paths or a dict from each dataset's label to its path. Several files are analyzed as one dataset: their cells one after another over one set of genes, matched by name, each cell labelled with its dataset (the file name, or the dict's key) in `obs[batch_key]`. The matrix is the one `anndata.concat` builds (each row's entries sorted by gene), without the files being merged first. |
| `sparse` | bool | `True` | If True (default), return CSR for sparse files. |
| `lazy` | bool | `True` | If True (default), return instantly (metadata only). |
| `mode` | str | `'auto'` | Where the analysis runs. `"auto"` (default) lets the planning step choose from the file's shape and the budget: `"memory"` when the matrix fits, else `"out-of-core"`, in which the counts are copied once to the project folder and every step reads them in blocks. Either may be forced. The results are the same. |
| `name` | str \| None | `None` | The dataset's folder name in the project (default: the file name without its extension; for several files, their labels joined by `+`). |
| `batch_key` | str | `'dataset'` | For several files, the `obs` column of dataset labels. |
| `genes` | str | `'intersection'` | For several files, the genes kept: `"intersection"` (default; the genes every file has, in the first file's order) or `"union"` (every gene, zero where a file lacks it; sorted by name unless every file lists the same genes), as the inner and outer joins of `anndata.concat` order them. |
| `obs_columns` | list[str] \| None | `None` | The file's cell annotations to read into `obs` (default: all of them). Labels are kept as categoricals, one small integer per cell. For several files, a column one file lacks is missing for its cells. |

**Returns:** A `LazyAnnData` (default) or `AnnData` if `lazy=False`.

### `ps.read_10x_h5`

```python
ps.read_10x_h5(path: str, *, lazy: bool = True) -> AnnData | LazyAnnData
```

Read a 10x Genomics `.h5` file (Cell Ranger format).

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str | required | Path to the `filtered_feature_bc_matrix.h5` file. |
| `lazy` | bool | `True` | If True (default), return instantly with metadata only. The data matrix materializes when first accessed. Set False to load everything eagerly. |

**Returns:** A `LazyAnnData` (default) or `AnnData` if `lazy=False`.

### `ps.read_10x_mtx`

```python
ps.read_10x_mtx(dir: str, *, lazy: bool = True) -> AnnData | LazyAnnData
```

Read a 10x MatrixMarket directory (`matrix.mtx` + `barcodes.tsv` + `features.tsv` or `genes.tsv`).

| argument | type | default | meaning |
|---|---|---|---|
| `dir` | str | required | Directory containing the three files. |
| `lazy` | bool | `True` | If True (default), return instantly with metadata only. |

**Returns:** A `LazyAnnData` (default) or `AnnData` if `lazy=False`.

### `ps.load`

```python
ps.load(project: str | os.PathLike, name: str, *, relink: dict[str, str] | None = None) -> Any
```

Open an analysis saved with `adata.save(name)`, to continue where it stopped.

| argument | type | default | meaning |
|---|---|---|---|
| `project` | str \| os.PathLike | required | The project folder the analysis was saved in. |
| `name` | str | required | The saved analysis's name. |
| `relink` | dict[str, str] \| None | `None` | New places of moved source files, `{old path: new path}`. |

**Raises:** SourceMovedError: A source file is no longer where it was; the message says where it was expected.

### `ps.process_diskbacked`

```python
ps.process_diskbacked(path: str, *, n_hvg: int = 2000, n_pcs: int = 50, n_neighbors: int = 15, target_sum: float = 10000.0, resolution: float = 1.0, seed: int = 42, min_genes: int = 0, max_genes: float = inf, min_counts: float = 0, max_counts: float = inf, max_pct_mt: float = inf, mito_prefix: str = 'MT-', min_cells: int = 0, include: list[str] | None = None) -> AnnData
```

Run the pipeline on a dataset that exceeds RAM, using SSD.

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str | required | Path to the `.h5ad` file. |
| `n_hvg` | int | `2000` | Number of highly-variable genes. |
| `n_pcs` | int | `50` | Number of principal components. |
| `n_neighbors` | int | `15` | Neighbours per cell for the kNN graph. |
| `target_sum` | float | `10000.0` | Target counts per cell for normalisation. |
| `resolution` | float | `1.0` | Leiden resolution. |
| `seed` | int | `42` | Random seed. |
| `min_genes` | int | `0` | Minimum detected genes for a cell to be kept (0 keeps all cells), matching `ps.pp.filter_cells`. |
| `max_genes` | float | `inf` | Maximum detected genes for a cell to be kept. |
| `min_counts` | float | `0` | Minimum total counts for a cell to be kept. |
| `max_counts` | float | `inf` | Maximum total counts for a cell to be kept. |
| `max_pct_mt` | float | `inf` | Maximum percentage of a cell's counts from mitochondrial genes (`obs['pct_counts_mt']`, reported whenever such genes exist). |
| `mito_prefix` | str | `'MT-'` | Prefix of the mitochondrial gene names, as in `ps.pp.calculate_qc_metrics`. |
| `min_cells` | int | `0` | Minimum number of kept cells in which a gene is detected for it to be kept, matching `ps.pp.filter_genes` after `filter_cells`; removed genes take no part in normalization, gene selection or PCA (`var['kept']`). |
| `include` | list[str] \| None | `None` | Genes to keep whatever the ranking chooses, added to the top `n_hvg` (see `ps.pp.highly_variable_genes`). A name not in the file, or a gene the `min_cells` filter removes, is an error naming it, raised before the gene selection runs. |

**Returns:** An `AnnData` carrying `obsm['X_pca']`, the neighbour graph, cluster labels, and the QC/HVG results in `obs`/`var`.

### `ps.set_workflow`

```python
ps.set_workflow(name: str) -> None
```

Choose the defaults of the analysis steps.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | `'polariseq'` (default) or `'scanpy'`. |

### `ps.workflow`

```python
ps.workflow() -> str
```

The active workflow (see `set_workflow`).

### `ps.get_build_info`

```python
ps.get_build_info() -> dict[str, Any]
```

Return a dict describing the linked Rust core.

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

An independent copy (out of core, of the view, not the counts).

### `adata.from_counts`

```python
adata.from_counts(name: str, *, genes: str = 'all') -> AnnData
```

A new analysis of the same cells, starting again from their raw counts.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | required | The new analysis's name (letters, digits, `. _ + -`). |
| `genes` | str | `'all'` | `"all"` (default): every gene of the source files, to filter and select again; `"current"`: this analysis's genes. |

**Returns:** The new analysis; this one is unchanged.

### `adata.save`

```python
adata.save(name: str, *, overwrite: bool = False) -> Any
```

Save the analysis in the project folder, to continue it later with `load`.

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

### `adata.write_h5ad`

```python
adata.write_h5ad(path: str | os.PathLike, *, counts: bool = False) -> Any
```

Write the analysis as an `.h5ad` file for other tools (Scanpy, cellxgene, R).

| argument | type | default | meaning |
|---|---|---|---|
| `path` | str \| os.PathLike | required | The `.h5ad` file to write (replaced if it exists). |
| `counts` | bool | `False` | Also write the raw counts as `layers['counts']`. |

**Returns:** The path written.

### `adata.head`

```python
adata.head(n: int = 10) -> dict
```

Preview the first n cells — partial read, fast.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | `10` | Number of cells, from the first. |

**Returns:** A dict with `data`, `indices`, `indptr`, `n_cells` and `n_genes` (the cells' raw values).

### `adata.sample`

```python
adata.sample(n: int = 100, seed: int = 42)
```

Preview a random sample of cells as a scipy CSR matrix.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | `100` | Number of cells, drawn at random. |
| `seed` | int | `42` | Random seed. |

**Returns:** A SciPy CSR matrix of the cells' raw values.

### `adata.validate`

```python
adata.validate() -> dict[str, Any]
```

Check data integrity without materializing. Returns a report.

### `adata.close`

```python
adata.close() -> None
```

Release all resources: free matrix memory, close mmaps, delete temp files.

## `ps.pp`

Preprocessing, in the usual order: quality control, filters, doublets (`scrublet`, on raw counts),
normalization, gene selection, PCA, integration (`harmony_integrate`), the neighbour graph.

### `ps.pp.calculate_qc_metrics`

```python
ps.pp.calculate_qc_metrics(adata: AnnData, *, mito_prefix: str = 'MT-') -> None
```

Compute per-cell and per-gene quality control metrics.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `mito_prefix` | str | `'MT-'` | Prefix for mitochondrial gene names (default "MT-"). |

### `ps.pp.filter_cells`

```python
ps.pp.filter_cells(adata: AnnData, *, min_counts: float = 0, max_counts: float = inf, min_genes: int = 0, max_genes: int = inf, max_pct_mt: float = inf, inplace: bool = True) -> AnnData | None
```

Filter cells based on QC metrics.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `min_counts` | float | `0` | Minimum total counts per cell. |
| `max_counts` | float | `inf` | Maximum total counts per cell. |
| `min_genes` | int | `0` | Minimum detected genes per cell. |
| `max_genes` | int | `inf` | Maximum detected genes per cell. |
| `max_pct_mt` | float | `inf` | Maximum percentage of counts from mitochondrial genes (`obs['pct_counts_mt']`; needs mitochondrial genes in the data). |
| `inplace` | bool | `True` | If True (default), modify in place. Else return filtered copy. |

**Returns:** None if inplace, else a new filtered AnnData.

### `ps.pp.filter_genes`

```python
ps.pp.filter_genes(adata: AnnData, *, min_cells: int = 0, min_counts: float = 0, inplace: bool = True) -> AnnData | None
```

Filter genes based on detection across cells.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `min_cells` | int | `0` | Minimum number of cells expressing the gene. |
| `min_counts` | float | `0` | Minimum total counts for the gene. |
| `inplace` | bool | `True` | If True (default), modify in place. |

**Returns:** None if inplace, else a new filtered AnnData.

### `ps.pp.harmony_integrate`

```python
ps.pp.harmony_integrate(adata: AnnData, batch_key: str | list[str], *, use_rep: str = 'X_pca', key_added: str = 'X_pca_harmony', theta: float | list[float] = 2.0, sigma: float = 0.1, lamb: float | list[float] | None = None, n_clusters: int | None = None, tau: float = 0.0, block_size: float = 0.05, max_iter_harmony: int = 10, max_iter_kmeans: int = 4, epsilon_cluster: float = 0.001, epsilon_harmony: float = 0.01, alpha: float = 0.2, seed: int = 0) -> None
```

Correct the PCA embedding for batch effects with Harmony.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `batch_key` | str \| list[str] | required | Column of `adata.obs` naming each cell's batch, or a list of columns to correct for together (e.g. `["donor", "lane"]`). |
| `use_rep` | str | `'X_pca'` | The `obsm` embedding to correct. |
| `key_added` | str | `'X_pca_harmony'` | The `obsm` key of the corrected embedding. |
| `theta` | float \| list[float] | `2.0` | Diversity penalty, one for all batch variables or one each. Larger values push harder for clusters that mix the batches. |
| `sigma` | float | `0.1` | Width of the soft clustering; smaller is closer to hard k-means. |
| `lamb` | float \| list[float] \| None | `None` | Ridge penalty, one for all variables or one each. `None` estimates it per cluster from the expected number of cells (`alpha` times it), as harmonypy 2 does. |
| `n_clusters` | int \| None | `None` | Number of clusters; default `min(n_cells / 30, 100)`. |
| `tau` | float | `0.0` | Protection of small batches against overcorrection (0 = off). |
| `block_size` | float | `0.05` | Fraction of the cells reassigned at a time. |
| `max_iter_harmony` | int | `10` | Maximum rounds of clustering and correction. |
| `max_iter_kmeans` | int | `4` | Assignment rounds per clustering. |
| `epsilon_cluster` | float | `0.001` | Convergence threshold of the assignment rounds. |
| `epsilon_harmony` | float | `0.01` | Convergence threshold of the rounds of clustering and correction. |
| `alpha` | float | `0.2` | Scale of the estimated ridge penalty. |
| `seed` | int | `0` | Random seed. |

### `ps.pp.highly_variable_genes`

```python
ps.pp.highly_variable_genes(adata: AnnData, n_top_genes: int = 2000, *, flavor: str | None = None, subset: bool = False, min_cells: int | None = None, n_bins: int | None = None, lowess_frac: float = 0.1, include: list[str] | None = None) -> None
```

Select highly-variable genes (scanpy `pp.highly_variable_genes`).

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `n_top_genes` | int | `2000` | Number of top HVGs to select (exact count). |
| `flavor` | str \| None | `None` | `'seurat'` or `'scarf'`. Left out, the workflow decides (`set_workflow`): `'polariseq'` ranks as `'seurat'` among the genes detected in more than 1% of the cells, `'scanpy'` is `'seurat'` over every gene. |
| `subset` | bool | `False` | If True, subset the matrix to HVGs (updates X, var, var_names). |
| `min_cells` | int \| None | `None` | a gene must be detected in more cells than this. For `'scarf'` the default is `int(0.01 * n_obs)`; for `'seurat'` the default is no floor (scanpy's behaviour), and a floor ranks only the genes above it, as if the others were not in the data. |
| `n_bins` | int \| None | `None` | Mean-expression bins (default 20 for `'seurat'`, 200 for `'scarf'`). |
| `lowess_frac` | float | `0.1` | `'scarf'` only: the share of floor genes in each local fit of the trend. |
| `include` | list[str] \| None | `None` | Genes to keep whatever the ranking chooses, by name: those not among the top `n_top_genes` are added to them (so up to `n_top_genes + len(include)` genes are kept), past the detection floor. `var['included']` marks them and `uns['hvg']['include']` records the request. With `subset=True` the renormalization and PCA that follow run on them too; note that `pca(scale=True)` gives each gene equal weight, a rarely detected one included. A name that is not among the current genes is an error saying why (absent from the data, or removed by an earlier step) and the closest names. |

### `ps.pp.log1p`

```python
ps.pp.log1p(adata: AnnData | LazyAnnData) -> None
```

Apply `log1p` (natural log of 1+x) to `adata.X` in place.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |

### `ps.pp.neighbors`

```python
ps.pp.neighbors(adata: AnnData, n_neighbors: int = 15, *, use_rep: str = 'X_pca', n_pcs: int | None = None, metric: str = 'euclidean', approximate: bool | None = None, seed: int = 42) -> None
```

Build a kNN graph from an embedding, storing it in `adata.obsp`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `n_neighbors` | int | `15` | Number of nearest neighbors per cell. |
| `use_rep` | str | `'X_pca'` | Which `obsm` key to use as the embedding. |
| `n_pcs` | int \| None | `None` | Number of PCs to use (default: all available). |
| `metric` | str | `'euclidean'` | Distance metric ('euclidean' or 'cosine'). |
| `approximate` | bool \| None | `None` | Force exact (False) or approximate (True) kNN. Default: auto (exact ≤ 10K cells, approximate above). |
| `seed` | int | `42` | Random seed for NN-Descent. |

### `ps.pp.normalize_total`

```python
ps.pp.normalize_total(adata: AnnData, target_sum: float = 10000.0) -> None
```

Scale each cell's total counts to `target_sum` (scanpy `pp.normalize_total`).

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `target_sum` | float | `10000.0` | Target sum per cell (default 1e4). |

### `ps.pp.pca`

```python
ps.pp.pca(adata: AnnData, n_components: int = 50, *, center: bool = True, scale: bool = False, max_value: float | None = None, seed: int = 0) -> None
```

Run PCA on `adata.X`, storing the embedding in `adata.obsm['X_pca']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `n_components` | int | `50` | Number of principal components. |
| `center` | bool | `True` | Whether to center columns (scanpy zero_center default). |
| `scale` | bool | `False` | Standardize each gene first, as `sc.pp.scale` before `sc.tl.pca`. The scaled matrix is never formed, so a sparse `X` stays sparse and unchanged. Implies centering. |
| `max_value` | float \| None | `None` | With `scale`, clip scaled values to `±max_value` (scanpy's `max_value`). |
| `seed` | int | `0` | Random seed for the randomized path. |

### `ps.pp.renormalize`

```python
ps.pp.renormalize(adata: AnnData, target_sum: float = 1000.0, *, log1p: bool = True, counts: AnnData | None = None) -> None
```

Normalize the current genes' raw counts again, over those genes only.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `target_sum` | float | `1000.0` | Total each cell is scaled to. |
| `log1p` | bool | `True` | Log-transform after scaling. |
| `counts` | AnnData \| None | `None` | In memory, the raw counts when `adata` is already normalized. |

### `ps.pp.scrublet`

```python
ps.pp.scrublet(adata: AnnData | LazyAnnData, *, batch_key: str | None = None, sim_doublet_ratio: float = 2.0, expected_doublet_rate: float = 0.05, stdev_doublet_rate: float = 0.02, n_neighbors: int | None = None, n_prin_comps: int = 30, min_counts: float = 3, min_cells: int = 3, min_gene_variability_pctl: float = 85.0, threshold: float | None = None, random_state: int = 0) -> None
```

Score each cell's chance of being a doublet, and call doublets.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The data (mutated in place). |
| `batch_key` | str \| None | `None` | `obs` column of the capture each cell came from. |
| `sim_doublet_ratio` | float | `2.0` | Doublets simulated per cell. |
| `expected_doublet_rate` | float | `0.05` | Expected share of doublets. |
| `stdev_doublet_rate` | float | `0.02` | Uncertainty of that share. |
| `n_neighbors` | int \| None | `None` | Neighbours per cell (default `round(0.5 √n)`). |
| `n_prin_comps` | int | `30` | Principal components of the embedding. |
| `min_counts` | float | `3` | A gene is used if it reaches `min_counts` normalized counts in `min_cells` cells. |
| `min_cells` | int | `3` | A gene is used if it reaches `min_counts` normalized counts in `min_cells` cells. |
| `min_gene_variability_pctl` | float | `85.0` | Genes are used above this percentile of variability (the v-score). |
| `threshold` | float \| None | `None` | Score above which a cell is called a doublet. |
| `random_state` | int | `0` | Seed of the simulation. |

## `ps.tl`

Tools on the neighbour graph and the clusters: `leiden`, `umap`, markers (`rank_genes_groups`),
annotation and curation, `paris`, `paga`, `dpt`, and comparisons of conditions
(`compare_conditions`) and of analyses (`compare_analyses`).

### `ps.tl.annotate`

```python
ps.tl.annotate(adata: AnnData, mapping: dict, groupby: str = 'leiden', *, key_added: str = 'cell_type', unassigned: str | None = None) -> None
```

Name clusters, `{"0": "CD4 T cell"}` or `{"B cell": ["4", "9"]}`; clusters given one name are joined. See `polariseq._curation.annotate`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (in memory or out of core). |
| `mapping` | dict | required | `{cluster: name}` or `{name: [clusters]}`. |
| `groupby` | str | `'leiden'` | The label column the clusters come from. |
| `key_added` | str | `'cell_type'` | The column the names are written to. |
| `unassigned` | str \| None | `None` | The name of clusters left out of `mapping` (default: their own label). |

### `ps.tl.auto_annotate`

```python
ps.tl.auto_annotate(adata: AnnData, groupby: str = 'leiden', *, tissue: str | None = None, species: str = 'human', method: str = 'panglaodb') -> None
```

Automatically assign cell type labels to clusters.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `groupby` | str | `'leiden'` | Column in obs identifying clusters. |
| `tissue` | str \| None | `None` | Restrict to one tissue's markers: `'immune'` (or `'pbmc'`, `'blood'`), `'brain'`, `'heart'`, `'lung'`, `'kidney'`, `'pancreas'`, `'liver'`, `'skin'`, `'gut'`, `'reproductive'`, or with `'panglaodb'` any PanglaoDB organ name (`'Thymus'`, `'Eye'`, ...). An unknown tissue is an error. |
| `species` | str | `'human'` | "human" or "mouse" (used by PanglaoDB). |
| `method` | str | `'panglaodb'` | Annotation method ('panglaodb', 'markers', or 'celltypist'). |

### `ps.tl.compare_analyses`

```python
ps.tl.compare_analyses(analyses: dict, key: str = 'leiden', *, labels: str | None = None, seeds: int = 0, n_markers: int = 20) -> Any
```

How far analyses of the same cells agree: the effect of their settings.

| argument | type | default | meaning |
|---|---|---|---|
| `analyses` | dict | required | `{name: analysis}`, two or more. |
| `key` | str | `'leiden'` | The `obs` column of clusters in each (default `"leiden"`). |
| `labels` | str \| None | `None` | An `obs` column of reference labels (for example the authors' cell types) to score each analysis against. |
| `seeds` | int | `0` | Rerun each analysis's Leiden clustering with this many other seeds and report its agreement with itself: the run-to-run variation a difference between settings should exceed. The analyses are left as they were. |
| `n_markers` | int | `20` | Top marker genes compared between matched clusters, when both analyses ran `rank_genes_groups` on `key`. |

**Returns:** An `polariseq._analyses.AnalysisComparison`: `ari` and `nmi` (every pair), `n_clusters`, `matches` and `crossing` (per pair), `labels` and `noise` when asked; printing it shows a summary. `ps.pl.compare_analyses` draws the clusters side by side.

### `ps.tl.compare_conditions`

```python
ps.tl.compare_conditions(adata: AnnData | LazyAnnData, condition_key: str, reference: str, *, groupby: str | None = None, sample_key: str | None = None, conditions: list[str] | None = None, groups: list[str] | None = None, min_cells: int = 10, key_added: str = 'compare_conditions') -> Any
```

Compare conditions within each cell type: which genes change, and how much.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The data. `uns[key_added]` receives the table, one summary per cell type and the parameters. |
| `condition_key` | str | required | `obs` column of the conditions. |
| `reference` | str | required | The condition the others are compared with. |
| `groupby` | str \| None | `None` | `obs` column of the cell types (default: all cells as one group). |
| `sample_key` | str \| None | `None` | `obs` column of the biological replicates. |
| `conditions` | list[str] \| None | `None` | Conditions to compare (default: all but the reference). |
| `groups` | list[str] \| None | `None` | Cell types to compare (default: all). |
| `min_cells` | int | `10` | A replicate's cells in a cell type below which it takes no part (with `sample_key`); the cells each condition needs (without). |
| `key_added` | str | `'compare_conditions'` | Where in `uns` the results go. |

**Returns:** A `pandas.DataFrame`, one row per cell type, condition and gene tested: `log2_fold_change`, the mean expression of each side (log2 CPM for replicates, the normalized value for cells), the share of cells expressing the gene on each side, `t`, `p_value` and `adj_p_value` (Benjamini-Hochberg within each cell type and condition), most significant first.

### `ps.tl.dpt`

```python
ps.tl.dpt(adata: AnnData, root: int = 0) -> None
```

Compute diffusion pseudotime from a root cell.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `root` | int | `0` | Index of the root cell (pseudotime origin). |

### `ps.tl.leiden`

```python
ps.tl.leiden(adata: AnnData, resolution: float = 1.0, *, seed: int = 0, key_added: str = 'leiden', method: str = 'auto') -> None
```

Run community detection on the neighbor graph.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `resolution` | float | `1.0` | Higher gives more, smaller clusters; lower fewer, larger ones. |
| `seed` | int | `0` | Random seed; the same seed gives the same clusters. |
| `key_added` | str | `'leiden'` | The `obs` column the cluster numbers are written to. |
| `method` | str | `'auto'` | `'auto'` (Leiden), `'fast'` (multi-level Louvain) or `'igraph'` (igraph's Leiden, needs python-igraph). |

### `ps.tl.merge_clusters`

```python
ps.tl.merge_clusters(adata: AnnData, groups: list, groupby: str = 'leiden', *, key_added: str | None = None, names: list[str] | None = None) -> None
```

Join clusters into one, `[["3", "7"]]` becomes `"3+7"`. See `polariseq._curation.merge_clusters`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (in memory or out of core). |
| `groups` | list | required | The clusters to join, as a list of lists (a single list is one group). |
| `groupby` | str | `'leiden'` | The label column the clusters come from. |
| `key_added` | str \| None | `None` | The column written. |
| `names` | list[str] \| None | `None` | A name for each joined cluster. |

### `ps.tl.paga`

```python
ps.tl.paga(adata: AnnData, groupby: str = 'leiden') -> None
```

Compute PAGA (partition-based graph abstraction).

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `groupby` | str | `'leiden'` | Column in obs identifying clusters. |

### `ps.tl.paris`

```python
ps.tl.paris(adata: AnnData, n_clusters: int | None = None, *, balanced: bool = False, min_size: int | None = None, max_size: int | None = None, max_distance_fc: float = 2.0, key_added: str = 'paris', recompute: bool = False) -> None
```

Paris hierarchical clustering of the neighbour graph.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container, after `ps.pp.neighbors()`. |
| `n_clusters` | int \| None | `None` | Straight cut into this many clusters (or more, where merge heights tie). |
| `balanced` | bool | `False` | Use Scarf's balanced cut instead. |
| `min_size` | int \| None | `None` | Balanced cut: size above which heights are compared. |
| `max_size` | int \| None | `None` | Balanced cut: largest cluster. |
| `max_distance_fc` | float | `2.0` | Balanced cut: largest height ratio to join. |
| `key_added` | str | `'paris'` | `obs` / `uns` key. |
| `recompute` | bool | `False` | Rebuild the dendrogram even if one is stored. |

### `ps.tl.rank_genes_groups`

```python
ps.tl.rank_genes_groups(adata: AnnData, groupby: str = 'leiden', *, groups: list[str] | None = None, reference: str = 'rest', n_genes: int = 50) -> None
```

Rank genes by differential expression per group.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `groupby` | str | `'leiden'` | `obs` column of the groups: a clustering, or any labels (conditions, cell types). |
| `groups` | list[str] \| None | `None` | Groups to rank (default: all but the reference). |
| `reference` | str | `'rest'` | `"rest"`, or the group the others are compared with. |
| `n_genes` | int | `50` | Number of top genes to store per group. |

### `ps.tl.score_genes`

```python
ps.tl.score_genes(adata: AnnData, gene_list: list[str], score_name: str | None = None, *, seed: int = 42) -> None
```

Score cells by mean expression of a gene set.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container. |
| `gene_list` | list[str] | required | List of gene names to score. |
| `score_name` | str \| None | `None` | Column name for the score (default: auto-generated). |
| `seed` | int | `42` | Random seed for control set selection. |

### `ps.tl.subcluster`

```python
ps.tl.subcluster(adata: AnnData, cluster: Any, groupby: str = 'leiden', *, resolution: float = 1.0, seed: int = 0, key_added: str | None = None) -> int
```

Split one cluster by Leiden on its own cells, `"5"` becomes `"5.0"`, `"5.1"`, ...; returns the number of parts. See `polariseq._curation.subcluster`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `pp.neighbors`. |
| `cluster` | Any | required | The cluster to split. |
| `groupby` | str | `'leiden'` | The label column it belongs to. |
| `resolution` | float | `1.0` | Leiden resolution within the cluster (higher: more parts). |
| `seed` | int | `0` | Random seed. |
| `key_added` | str \| None | `None` | The column written (default `f"{groupby}_curated"`; `groupby` to replace it). |

### `ps.tl.umap`

```python
ps.tl.umap(adata: AnnData, *, min_dist: float | None = None, spread: float = 1.0, n_epochs: int = 0, seed: int = 42, deterministic: bool = False, init: str | np.ndarray | None = None, init_smooth: int = 30) -> None
```

Run UMAP on the neighbor graph, storing the 2D embedding in `adata.obsm['X_umap']`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The data container (mutated in place). |
| `min_dist` | float \| None | `None` | Minimum distance between embedded points (left out: 0.5 in the `'polariseq'` workflow, 0.1 in `'scanpy'`). |
| `spread` | float | `1.0` | Scale of embedded point spread. |
| `n_epochs` | int | `0` | SGD epochs (0 = auto: 500 up to 10,000 cells, else 100 in the `'polariseq'` workflow and 200 in `'scanpy'`). |
| `seed` | int | `42` | Random seed. |
| `deterministic` | bool | `False` | Run single-threaded, bit-reproducible SGD instead of the default parallel (Hogwild) mode. The parallel layout differs run to run by construction, as umap-learn's does. |
| `init` | str \| np.ndarray \| None | `None` | Where the cells start. `'spectral'` (the graph's leading eigenvectors, each axis in [-10, 10]), `'random'` (uniform in [-10, 10]), `'pca'` (the first two of `obsm['X_pca']`), `'kmeans'` (Scarf's start: 1,000 k-means centroids of `obsm['X_pca']` laid out by their own PCA, each cell at its centroid), `'multilevel'` (Polariseq's multilevel spectral start: the exact spectral layout of the graph between k-means groups of `obsm['X_pca']` (about 20 cells per group, 50 to 1,000 groups), lifted to the cells and refined by `init_smooth` smoothing steps on the cell graph), `'leiden'` (the same through the graph between the Leiden clusters in `obs['leiden']`), or an n × 2 array. Left out: `'multilevel'` in the `'polariseq'` workflow, `'spectral'` in `'scanpy'`. Any start but the first two is scaled, axis by axis, to [0, 10], umap-learn's convention for every start. |
| `init_smooth` | int | `30` | Smoothing steps of the multilevel starts. |

## `ps.get`

Tables taken out of an analysis.

### `ps.get.markers`

```python
ps.get.markers(adata: AnnData, n: int = 10, *, groups: list | None = None, wide: bool = False) -> Any
```

Top marker genes per cluster from `tl.rank_genes_groups`, as a table (`wide=True`: one column of gene names per cluster). See `polariseq._curation.markers`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis, after `tl.rank_genes_groups`. |
| `n` | int | `10` | Genes per cluster. |
| `groups` | list \| None | `None` | Clusters to include (default: all ranked). |
| `wide` | bool | `False` | One column of gene names per cluster instead of the long table. |

### `ps.get.pseudobulk`

```python
ps.get.pseudobulk(adata: AnnData | LazyAnnData, sample_key: str, *, groupby: str | None = None, condition_key: str | None = None, min_cells: int = 0) -> AnnData
```

Raw counts summed per replicate (and per cell type and condition).

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData \| LazyAnnData | required | The analysis; the raw counts are read wherever they are held. |
| `sample_key` | str | required | `obs` column of the biological replicates (for example the donor). |
| `groupby` | str \| None | `None` | `obs` column of the cell types (default: all cells as one group). |
| `condition_key` | str \| None | `None` | `obs` column of the conditions, to keep them apart. |
| `min_cells` | int | `0` | A combination with fewer cells is left out. |

**Returns:** An object with one row per combination (`obs` names its groupby, condition and replicate, and `obs['n_cells']`), the summed counts in `X`, the genes in `var_names`.

## `ps.pl`

Plots. Every function takes `show=` and `save=` (a file path); most take `ax=` to draw into an
existing Matplotlib axis.

### `ps.pl.categorical_palette`

```python
ps.pl.categorical_palette(n: int, washed: bool = True) -> list[str]
```

Palette for n classes: washed Tol, or golden-angle HSL beyond 22.

| argument | type | default | meaning |
|---|---|---|---|
| `n` | int | required | Number of colours. |
| `washed` | bool | `True` | Soften the colours (the theme's wash) so many groups stay calm; `False` gives the full colours. |

**Returns:** A list of hex colours.

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

### `ps.pl.compare_analyses`

```python
ps.pl.compare_analyses(analyses: dict, key: str = 'leiden', *, basis: str = 'X_umap', ncols: int | None = None, figsize: tuple | None = None, dpi: int | None = None, show: bool = True, save: str | None = None, **kw: Any) -> Any
```

Each analysis's clusters side by side, on one embedding.

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

### `ps.pl.dotplot`

```python
ps.pl.dotplot(adata: AnnData, markers: list[str] | None = None, groupby: str = 'leiden', ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Dot plot: size = fraction expressing, colour = mean expression.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `markers` | list[str] \| None | `None` | Genes to show (or `var_names=`), one column each. |
| `groupby` | str | `'leiden'` | `obs` column of the groups, one row each. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.embedding`

```python
ps.pl.embedding(adata: AnnData, basis: str = 'X_umap', color: str | list[str] | None = None, **kw) -> Any
```

Scatter any 2-D embedding in `obsm`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `basis` | str | `'X_umap'` | The `obsm` embedding (`'X_umap'`, `'X_pca'`, ...). |
| `color` | str \| list[str] \| None | `None` | An `obs` column or a gene; a list draws one panel each. |

### `ps.pl.fold_changes`

```python
ps.pl.fold_changes(result: Any, genes: list[str] | None = None, *, condition: str | None = None, n_genes: int = 6, alpha: float = 0.05, ax: Any | None = None, show: bool = True, save: str | None = None, key: str = 'compare_conditions', **kw) -> Any
```

Which cell types respond: log2 fold changes, genes against cell types.

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

### `ps.pl.get_theme`

```python
ps.pl.get_theme() -> dict
```

### `ps.pl.heatmap`

```python
ps.pl.heatmap(adata: AnnData, markers: list[str] | None = None, groupby: str = 'leiden', ax: Any | None = None, cmap: Any = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Heatmap of z-scored marker expression across groups.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `markers` | list[str] \| None | `None` | Genes to show (or `var_names=`). |
| `groupby` | str | `'leiden'` | `obs` column of the groups. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `cmap` | Any | `None` | Matplotlib colour map (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.highly_variable_genes`

```python
ps.pl.highly_variable_genes(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Mean–variance relation with HVGs softly highlighted.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.overview`

```python
ps.pl.overview(adata: AnnData, out: str, *, cluster_key: str = 'leiden', color_by: list[str] | tuple = (), thresholds: dict | None = None, qc_before: dict | None = None, prefix: str = '') -> dict
```

The standard figures of an analysis, written to `out`, each drawn the way that suits the size of the data: quality-control distributions over every cell, variance explained by the principal components, cluster sizes, and the UMAP coloured by cluster and by any obs column in `color_by` (drawn as density above :data:`DENSITY_ABOVE` cells). `qc_before` is the per-cell metrics from before filtering (see `qc_distributions`), so the cells a threshold removed are shown. Also writes `figure_data.json`, the numbers behind the quality-control, cluster-size and variance plots (`overview_data`). Returns `{figure: path}` for the figures that could be drawn.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `out` | str | required | Folder the figures are written to (created if missing). |
| `cluster_key` | str | `'leiden'` | `obs` column of the clusters. |
| `color_by` | list[str] \| tuple | `()` | Further `obs` columns to colour the UMAP by, one figure each. |
| `thresholds` | dict \| None | `None` | Filter thresholds to mark, `{'n_genes': 200}`. |
| `qc_before` | dict \| None | `None` | Per-cell metrics from before filtering, to show what was removed. |
| `prefix` | str | `''` | Text put before each file name. |

**Returns:** `{figure: path}` for each figure written.

### `ps.pl.overview_data`

```python
ps.pl.overview_data(adata: AnnData, *, cluster_key: str = 'leiden', qc_before: dict | None = None, thresholds: dict | None = None, bins: int = 64) -> dict
```

The numbers behind the figures of `overview`, as plain Python values.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `cluster_key` | str | `'leiden'` | `obs` column of the clusters. |
| `qc_before` | dict \| None | `None` | Per-cell metrics from before filtering. |
| `thresholds` | dict \| None | `None` | Filter thresholds, `{'n_genes': 200}`. |
| `bins` | int | `64` | Histogram bins (log-spaced). |

### `ps.pl.pca`

```python
ps.pl.pca(adata: AnnData, color: str | None = None, ax: Any | None = None, size: float | None = None, title: str | None = None, cmap: str | None = None, show: bool = True, save: str | None = None, legend_loc: str | None = 'right', **kw) -> Any
```

PCA scatter (PC1 vs PC2). Same extra attributes as `umap`.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `color` | str \| None | `None` | An `obs` column or a gene to colour the cells by. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `size` | float \| None | `None` | Point size (default: chosen from the number of cells). |
| `title` | str \| None | `None` | Title above the plot (default: what is shown). |
| `cmap` | str \| None | `None` | Matplotlib colour map for values (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `legend_loc` | str \| None | `'right'` | `'right'`, `'on data'` (labels at the groups) or `None`. |

### `ps.pl.pca_variance_ratio`

```python
ps.pl.pca_variance_ratio(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, n_comps: int = 30, **kw) -> Any
```

Variance explained per PC + cumulative curve on one calm axis.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `n_comps` | int | `30` | Principal components shown. |

### `ps.pl.qc_distributions`

```python
ps.pl.qc_distributions(adata: AnnData, keys: tuple | list = ('n_genes', 'n_counts'), *, thresholds: dict | None = None, before: dict | None = None, bins: int = 64, show: bool = True, save: str | None = None, figsize: tuple | None = None, dpi: int | None = None, **kw) -> Any
```

Distributions of per-cell quality metrics, from every cell.

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

### `ps.pl.rank_genes_groups`

```python
ps.pl.rank_genes_groups(adata: AnnData, groups: list[str] | None = None, n_genes: int = 10, ax: Any | None = None, show: bool = True, save: str | None = None, *, ncols: int | None = None, figsize: tuple | None = None, dpi: int | None = None, sharey: bool = False, key: str = 'rank_genes_groups', **kw) -> Any
```

Top marker genes per cluster, ranked, one panel per group.

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

### `ps.pl.scatter_qc`

```python
ps.pl.scatter_qc(adata: AnnData, ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

QC scatter: genes vs counts with density rendering at scale.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.scrublet_score_distribution`

```python
ps.pl.scrublet_score_distribution(adata: AnnData, *, log: bool = True, show: bool = True, save: str | None = None, **kw) -> Any
```

Doublet scores of the cells and of the simulated doublets, per batch.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `log` | bool | `True` | Logarithmic count axis for the cells. |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.set_theme`

```python
ps.pl.set_theme(name: str = 'paper', **overrides: Any) -> None
```

Set the global theme, optionally overriding individual keys.

| argument | type | default | meaning |
|---|---|---|---|
| `name` | str | `'paper'` | `'paper'`, `'notebook'`, `'talk'`, `'dark'`, `'print'` or `'classic'`. |

### `ps.pl.stacked_bar`

```python
ps.pl.stacked_bar(adata: AnnData, groupby: str = 'leiden', color_by: str = 'dataset', ax: Any | None = None, show: bool = True, save: str | None = None, **kw) -> Any
```

Horizontal 100% composition bars with legend below.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `groupby` | str | `'leiden'` | `obs` column of the bars (for example the clusters). |
| `color_by` | str | `'dataset'` | `obs` column split within each bar (for example the dataset or condition). |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |

### `ps.pl.umap`

```python
ps.pl.umap(adata: AnnData, color: str | None = None, ax: Any | None = None, size: float | None = None, title: str | None = None, cmap: str | None = None, show: bool = True, save: str | None = None, legend_loc: str | None = 'right', **kw) -> Any
```

UMAP scatter — calm palette, density-ordered, optional on-data labels.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `color` | str \| None | `None` | An `obs` column or a gene to colour the cells by (a list draws one panel each); `None` draws the cells in grey. |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `size` | float \| None | `None` | Point size (default: chosen from the number of cells). |
| `title` | str \| None | `None` | Title above the plot (default: what is shown). |
| `cmap` | str \| None | `None` | Matplotlib colour map for values (default: the theme's). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `legend_loc` | str \| None | `'right'` | `'right'`, `'on data'` (labels at the groups) or `None`. |

### `ps.pl.violin`

```python
ps.pl.violin(adata: AnnData, keys: list[str], groupby: str | None = None, ax: Any | None = None, show: bool = True, save: str | None = None, *, split_by: str | None = None, **kw) -> Any
```

Soft violins with median dots; calm palette; rich text attributes.

| argument | type | default | meaning |
|---|---|---|---|
| `adata` | AnnData | required | The analysis (what `ps.read_h5ad` returns); steps add their results to it in place. |
| `keys` | list[str] | required | `obs` columns or genes, one violin set each. |
| `groupby` | str \| None | `None` | `obs` column of the groups (default: all cells as one). |
| `ax` | Any \| None | `None` | A Matplotlib axis to draw into (default: a new figure). |
| `show` | bool | `True` | Show the figure (`plt.show()`); `False` to keep drawing on it or only save it. |
| `save` | str \| None | `None` | A file to write the figure to; the extension sets the format (`.png`, `.pdf`, `.svg`). |
| `split_by` | str \| None | `None` | `obs` column of two levels (for example the condition), drawn as the two halves of each violin. |

### `ps.pl.volcano`

```python
ps.pl.volcano(result: Any, group: str | None = None, *, condition: str | None = None, n_labels: int = 10, alpha: float = 0.05, min_log2_fold_change: float = 1.0, ax: Any | None = None, show: bool = True, save: str | None = None, key: str = 'compare_conditions', **kw) -> Any
```

Fold change against significance for one cell type's comparison.

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

### `AnalysisComparison`

What `ps.tl.compare_analyses` returns.

| attribute | meaning |
|---|---|
| `key` | The `obs` column of clusters compared. |

### `polariseq.markers.get_all_markers`

```python
polariseq.markers.get_all_markers() -> dict[str, list[str]]
```

Get all markers from all tissues (flattened).

### `polariseq.markers.get_tissue_markers`

```python
polariseq.markers.get_tissue_markers(tissue: str) -> dict[str, list[str]]
```

Get marker genes for a specific tissue/system.

| argument | type | default | meaning |
|---|---|---|---|
| `tissue` | str | required | Tissue name (e.g., 'immune', 'heart', 'brain'). Aliases like 'pbmc', 'blood', 'cardiac' are accepted. |

**Returns:** Dict mapping cell type name → list of marker gene symbols.

### `polariseq.markers.score_cell_type`

```python
polariseq.markers.score_cell_type(gene_scores: dict[str, float], markers: list[str]) -> float
```

Score a cell type match given per-gene scores.

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
