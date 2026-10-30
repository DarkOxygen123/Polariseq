# Correct for batches

Run Harmony on the principal components, then build the graph on the corrected embedding:

```python
import polariseq as ps

adata = ps.read_h5ad(["batch1.h5ad", "batch2.h5ad"])
# quality control, normalization and gene selection as usual, then:
ps.pp.pca(adata, n_components=50, scale=True)
ps.pp.harmony_integrate(adata, "dataset")           # adds obsm["X_pca_harmony"]
ps.pp.neighbors(adata, use_rep="X_pca_harmony")
ps.tl.leiden(adata)
ps.tl.umap(adata)
```

Every later step that reads the graph uses the corrected embedding.

## Several batch variables

```python
ps.pp.harmony_integrate(adata, ["donor", "chemistry"], theta=[2.0, 1.0])
```

`theta` sets how strongly each variable is corrected (higher: more mixing).

## Check the result

Color the UMAP by batch and by cell type. Batches should mix within a cell type, and cell types should
stay apart:

```python
ps.pl.umap(adata, color=["dataset", "leiden"])
```

See also: [`ps.pp.harmony_integrate`](../api/pp/harmony_integrate.md),
[`ps.pp.neighbors`](../api/pp/neighbors.md), [Harmony](../METHODS.md#harmony).
