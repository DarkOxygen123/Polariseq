# Run the whole analysis in one call

`ps.process_diskbacked` runs quality control, the filters, normalization, gene selection, PCA, the
neighbor graph and Leiden clustering in one call, out of core.

```python
import polariseq as ps

ps.project("atlas_analysis")
ps.set_budget(ram_gb=6, threads=4)

adata = ps.process_diskbacked(
    "atlas.h5ad",
    min_genes=200, min_cells=3,
    n_hvg=2000, n_pcs=50, n_neighbors=15, resolution=1.0,
)
ps.tl.umap(adata)
ps.pl.umap(adata, color="leiden", show=False, save="umap.png")
```

It applies the same steps and defaults as the step-by-step code and analyzes the same cells and genes.

## Notes

- The filters keep every cell and gene unless you set them; set `min_genes` and `min_cells` as you would
  for `ps.pp.filter_cells` and `ps.pp.filter_genes`.
- More filters: `max_genes`, `min_counts`, `max_counts`, `max_pct_mt` (with `mito_prefix`).
- It uses `seed=42`; the step functions use `seed=0` by default. Pass `seed=0` to match them.

See also: [`ps.process_diskbacked`](../api/ps/process_diskbacked.md),
[Analyze a dataset larger than memory](large-data.md).
