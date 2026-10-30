# The analysis object

<figure markdown="span">
  ![The matrix X of cells by genes, with obs beside it for each cell, var above it for each gene, obsm embeddings and obsp graphs beside it, and uns](../assets/diagrams/anndata-light.svg#only-light)
  ![The matrix X of cells by genes, with obs beside it for each cell, var above it for each gene, obsm embeddings and obsp graphs beside it, and uns](../assets/diagrams/anndata-dark.svg#only-dark)
  <figcaption>The parts of an analysis.</figcaption>
</figure>

An analysis is one object, `adata`, shaped like Scanpy's AnnData so the same names mean the same
things. At its center is `X`, the matrix of cells by genes, sparse because most genes are not detected
in most cells. Everything else is aligned to it:

- `obs`: one row per cell, such as its quality metrics, cluster and cell type;
- `var`: one row per gene, such as its statistics and whether it was selected;
- `obsm`: per-cell embeddings, `X_pca` and `X_umap`;
- `obsp`: graphs between cells, the neighbor graph's `connectivities` and `distances`;
- `uns`: everything else, each step's parameters and results.

Each step reads the parts it needs and adds its results, in place: `ps.pp.pca` adds `obsm["X_pca"]`,
`ps.tl.leiden` adds `obs["leiden"]`. Subsetting, `adata[cells, genes]`, keeps every part aligned.

## Where the matrix lives

In memory, `X` is a sparse matrix held in RAM. Out of core, it is a view of the counts copied to the
project folder ([Out-of-core analysis](out-of-core.md)). `adata.mode` says which; the object looks the
same either way.

## Results on disk

With a project folder declared, large per-cell results (embeddings, graphs) are kept as memory-mapped
files in the folder, so they take little memory and are already on disk when the analysis is
[saved](../how-to/save-and-reopen.md).

See also: [The data object](../api/data/index.md), [Hand results to other tools](../how-to/export.md).
