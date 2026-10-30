# Compare conditions with donors as replicates

Compare a treated condition with its control within each cell type, with each donor as one replicate.
The analysis needs a condition column, a cell type column and a donor column in `obs`.

```python
import polariseq as ps

res = ps.tl.compare_conditions(adata, "condition", "control",
                               groupby="cell_type", sample_key="donor")
res.to_csv("treated_vs_control.csv", index=False)

significant = res[res["adj_p_value"] < 0.05]
print(significant.groupby("cell_type", observed=True).size())
```

The counts of each donor and cell type are summed (pseudobulk) and the conditions are tested across
donors. The result is one row per cell type and gene; `adata.uns["compare_conditions"]["groups"]` says
how many donors each cell type had and whether they were paired.

## Plot the result

```python
ps.pl.volcano(res, "CD14+ Monocytes")
ps.pl.fold_changes(res, n_genes=4)
ps.pl.violin(adata, ["ISG15", "IFI6"], groupby="cell_type", split_by="condition")
```

## Notes

- A cell type is tested only where each condition has donors with at least `min_cells` cells (default
  10).
- To export the summed counts for another tool, use
  `ps.get.pseudobulk(adata, "donor", groupby="cell_type", condition_key="condition")`.
- [`examples/05_comparing_conditions.py`](examples/05_comparing_conditions.md) runs this on the Kang et
  al. 2018 interferon-beta data, from download to figures.

See also: [`ps.tl.compare_conditions`](../api/tl/compare_conditions.md),
[`ps.get.pseudobulk`](../api/get/pseudobulk.md), [Pseudobulk testing](../METHODS.md#pseudobulk-testing).
