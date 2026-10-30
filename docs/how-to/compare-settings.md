# Try other settings and compare the results

Start several analyses of the same cells from the counts, each with its own settings, and measure how
far their clusters agree.

```python
import polariseq as ps

runs = {}
for n_genes, resolution in [(1000, 1.0), (2000, 1.0), (2000, 0.5)]:
    name = f"hvg{n_genes}_res{resolution}"
    b = adata.from_counts(name)                    # the same cells, from the counts
    ps.pp.normalize_total(b, target_sum=1e4)
    ps.pp.log1p(b)
    ps.pp.highly_variable_genes(b, n_top_genes=n_genes, subset=True)
    ps.pp.renormalize(b)
    ps.pp.pca(b, n_components=50, scale=True)
    ps.pp.neighbors(b)
    ps.tl.leiden(b, resolution=resolution)
    runs[name] = b

cmp = ps.tl.compare_analyses(runs, seeds=2)
print(cmp)
```

`ps.tl.compare_analyses` reports how far each pair of analyses agrees (adjusted Rand index). With
`seeds=2` it also reruns each analysis with other seeds: a difference between analyses smaller than the
seed-to-seed difference is noise.

## Keep genes of your own in the selection

```python
ps.pp.highly_variable_genes(b, n_top_genes=2000, subset=True, include=["CD8A", "FOXP3"])
```

## Notes

- Out of core, `from_counts` reuses the copy of the counts on disk, so a new analysis costs no extra
  disk.
- `cmp.matches[("hvg1000_res1.0", "hvg2000_res0.5")]` shows which cluster went where.
- `ps.pl.compare_analyses(runs)` draws the clusters of each analysis side by side on one embedding.

See also: [`adata.from_counts`](../api/data/from_counts.md),
[`ps.tl.compare_analyses`](../api/tl/compare_analyses.md),
[`examples/07_settings_saving_exporting.py`](examples/07_settings_saving_exporting.md).
