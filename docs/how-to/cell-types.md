# Name the cell types

## From marker databases

```python
import polariseq as ps

table = ps.tl.auto_annotate(adata, "leiden", tissue="immune")
print(table)
ps.pl.umap(adata, color="cell_type")
```

`ps.tl.auto_annotate` scores every cell for every cell type in the bundled marker tables and names each
cluster. It writes `obs["cell_type"]` and returns one row per cluster: the name and its Cell Ontology
term, the score, the share of the cluster's cells supporting it, the runner-up, and groups of cells
inside the cluster that score for another type.

- `tissue=` narrows the candidates; give it when you know the tissue (`"immune"`, `"brain"`, `"lung"`,
  ...).
- `source=` picks the marker database; `ps.marker_table(tissue="immune")` shows the markers that will be
  scored.

## By hand, from marker genes

```python
ps.tl.rank_genes_groups(adata, groupby="leiden")
print(ps.get.markers(adata, n=5, wide=True))       # the top genes of each cluster

ps.tl.annotate(adata, {"0": "CD4 T cells", "1": "CD14+ monocytes", "2": "B cells"})
```

Clusters left out of the mapping keep their number.

## Refine the clusters

```python
ps.tl.merge_clusters(adata, ["3", "7"])            # two clusters as one
ps.tl.subcluster(adata, "5", resolution=0.5)       # split cluster 5 into 5.0, 5.1, ...
```

Both write `obs["leiden_curated"]` and leave `obs["leiden"]` as it was.

See also: [`ps.tl.auto_annotate`](../api/tl/auto_annotate.md),
[`ps.marker_table`](../api/ps/marker_table.md), [`ps.tl.annotate`](../api/tl/annotate.md),
[`ps.get.markers`](../api/get/markers.md).
