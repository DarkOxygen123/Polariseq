# Analyze a dataset larger than memory

The analysis code is the same at any size. What you add is a project folder, a budget, and a look at
the plan before the data is read.

## 1. Declare the project folder and the budget

```python
import polariseq as ps

ps.project("/fast/disk/atlas_analysis")   # the counts are copied here; use a fast local disk
ps.set_budget(ram_gb=12, threads=8)       # the memory and threads Polariseq may use
```

## 2. Check the plan

```python
plan = ps.plan("atlas.h5ad")
print(plan.fits, plan.peak_gb, plan.limiting_step)
```

`ps.plan` reads only the file's description, so it returns at once even for a very large file. It
predicts the peak memory and disk of every step under the budget. If `plan.fits` is `False`, raise
`ram_gb`, free memory, or select fewer genes (`n_hvg=`).

## 3. Read the file and run the analysis

```python
adata = ps.read_h5ad("atlas.h5ad")
print(adata.mode)                         # "out-of-core" when the matrix does not fit

ps.pp.calculate_qc_metrics(adata)
ps.pp.filter_cells(adata, min_genes=200)
ps.pp.filter_genes(adata, min_cells=3)
ps.pp.normalize_total(adata, target_sum=1e4)
ps.pp.log1p(adata)
ps.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
ps.pp.renormalize(adata)
ps.pp.pca(adata, n_components=50, scale=True)
ps.pp.neighbors(adata, n_neighbors=15)
ps.tl.leiden(adata, resolution=1.0)
ps.tl.umap(adata)
```

On the first run the counts are copied into the project folder, at about one to two bytes per stored
value; later runs on the same file reuse the copy. `ps.project("/fast/disk/atlas_analysis").status()`
shows what the folder holds.

## Notes

- The out-of-core mode reads `.h5ad` files. Convert 10x files to `.h5ad` first
  (`ps.read_10x_h5(path).write_h5ad("data.h5ad")`).
- To force a mode, pass `mode="memory"` or `mode="out-of-core"` to `ps.read_h5ad`.
- For a run of millions of cells on a server, see [Run on a cluster node](run-on-a-cluster.md) and the
  [large-run script](examples/06_large_run.md).

See also: [`ps.plan`](../api/ps/plan.md), [`ps.read_h5ad`](../api/ps/read_h5ad.md),
[Memory and planning](../background/memory.md), [Out-of-core analysis](../background/out-of-core.md).
