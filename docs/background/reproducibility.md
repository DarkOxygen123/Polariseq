# Reproducible results

An analysis run twice should give the same result, on any computer and with any number of threads.

## Seeds

Every step with a random part takes a `seed`: the neighbor graph (`ps.pp.neighbors`), Leiden
(`ps.tl.leiden`), UMAP (`ps.tl.umap`), Scrublet (`random_state`) and Harmony. The same seed gives the
same result.

## Threads

<figure markdown="span">
  ![Six blocks of rows, each taken by one of three threads and giving a partial result; the partials are summed in block order](../assets/diagrams/rowblocks-light.svg#only-light)
  ![Six blocks of rows, each taken by one of three threads and giving a partial result; the partials are summed in block order](../assets/diagrams/rowblocks-dark.svg#only-dark)
  <figcaption>A pass over the cells, split among threads.</figcaption>
</figure>

A step that passes over the cells splits them into blocks and gives each block to a free thread. The
blocks finish in any order, but their partial results are combined in block order. Sums of decimal
numbers depend on the order they are added in, so a fixed order gives the same result at any number of
threads: the neighbor graph and the Leiden clusters are the same with 4 threads or 64.

UMAP's default layout runs in parallel and differs slightly from run to run, as umap-learn's does.
`ps.tl.umap(adata, deterministic=True)` runs it on one thread, giving the same layout every time.

## The budget

Out of core, the size of the blocks follows the memory budget. Pin it with `ps.set_budget(ram_gb=...)`
so that two runs read the data in the same blocks.

## The defaults

`ps.set_workflow("scanpy")` switches the defaults to Scanpy's own steps, for a step-by-step comparison
with Scanpy; `ps.workflow()` says which defaults are in force. Record it with the version
(`ps.__version__`) when you report an analysis.

See also: [Set the memory and threads](../how-to/memory-and-threads.md),
[Try other settings and compare the results](../how-to/compare-settings.md).
