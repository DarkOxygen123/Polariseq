# Remove doublets

Run Scrublet on the raw counts, before any normalization, once per capture:

```python
import numpy as np
import polariseq as ps

adata = ps.read_h5ad(["capture1.h5ad", "capture2.h5ad"])
ps.pp.scrublet(adata, batch_key="dataset")       # one capture per value of obs["dataset"]

ps.pl.scrublet_score_distribution(adata)         # check the threshold
adata = adata[~np.asarray(adata.obs["predicted_doublet"])]
```

`ps.pp.scrublet` adds `obs["doublet_score"]` and `obs["predicted_doublet"]`, and keeps each capture's
threshold and doublet rates in `uns["scrublet"]`.

## Check the threshold

In the score plot, the threshold (dashed) should sit in the valley between the two modes of the
simulated doublets. If it does not, set one:

```python
ps.pp.scrublet(adata, batch_key="dataset", threshold=0.25)
```

## Notes

- Doublets form within a capture, so score each capture separately with `batch_key`. Leave it out only
  for a single capture.
- `expected_doublet_rate` (default 0.05) is the share of doublets you expect from the loading density.

See also: [`ps.pp.scrublet`](../api/pp/scrublet.md),
[`ps.pl.scrublet_score_distribution`](../api/pl/scrublet_score_distribution.md),
[Scrublet](../METHODS.md#scrublet).
