# Read several files as one dataset

Give `ps.read_h5ad` a list of files, or a dict from a label to each file:

```python
import polariseq as ps

adata = ps.read_h5ad({"donor1": "d1.h5ad", "donor2": "d2.h5ad", "donor3": "d3.h5ad"})
print(adata.obs["dataset"])      # each cell's label
```

The cells of the files follow one another over one set of genes, matched by name. Each cell's label
(the dict key, or the file name for a list) is in `obs["dataset"]`; `batch_key=` names the column
something else.

## Choose the genes

- `genes="intersection"` (default): the genes every file has, in the first file's order.
- `genes="union"`: every gene, zero where a file lacks it.

## Plan them together

```python
plan = ps.plan(["d1.h5ad", "d2.h5ad", "d3.h5ad"])
```

Out of core, the files are copied once into one store in the project folder.

See also: [Correct for batches](batches.md), [`ps.read_h5ad`](../api/ps/read_h5ad.md),
[`ps.plan`](../api/ps/plan.md).
