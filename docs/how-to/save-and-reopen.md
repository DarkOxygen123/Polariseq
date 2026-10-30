# Save an analysis and continue later

```python
import polariseq as ps

ps.project("my_analysis")
# ... an analysis up to clustering ...
adata.save("clustered")
```

Later, in a new session:

```python
import polariseq as ps

adata = ps.load("my_analysis", "clustered")
ps.tl.umap(adata)
```

`ps.load` declares the project folder and returns the analysis as it was saved: its cells and genes,
the matrix as the steps left it, and every annotation, embedding, graph and result. It works the same
in memory and out of core.

## What is stored

The counts are not copied. A saved analysis points to the files the counts came from and links the
results the project folder already holds, so saving is quick and small.

## If a source file has moved

Loading names the file and where it was expected. Pass its new place:

```python
adata = ps.load("my_analysis", "clustered", relink={"/old/path/data.h5ad": "/new/path/data.h5ad"})
```

Keep the source files inside the project folder, and the two move together.

See also: [`adata.save`](../api/data/save.md), [`ps.load`](../api/ps/load.md),
[Hand results to other tools](export.md).
