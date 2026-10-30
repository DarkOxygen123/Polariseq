# Hand results to Scanpy and other tools

## Write an `.h5ad` file

```python
adata.write_h5ad("analysis.h5ad")                  # no anndata package needed
adata.write_h5ad("analysis.h5ad", counts=True)     # with the raw counts in layers["counts"]
```

The file opens in Scanpy, cellxgene and R. Out of core, it is written in blocks, so the
matrix never has to fit in memory.

## Hand the object to Scanpy

```python
import scanpy as sc

ad = adata.to_anndata()                            # needs pip install anndata
sc.pl.umap(ad, color="leiden")
```

`X` becomes a SciPy CSR matrix, and `obs`, `var`, `obsm`, `obsp` and `uns` are carried over.

## Tables

```python
ps.get.markers(adata, n=10).to_csv("markers.csv", index=False)
adata.obs["leiden"]                                 # per-cell results, by name
```

See also: [`adata.write_h5ad`](../api/data/write_h5ad.md),
[`adata.to_anndata`](../api/data/to_anndata.md), [Save and continue later](save-and-reopen.md).
