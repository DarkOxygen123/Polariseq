# Out-of-core analysis

An analysis runs out of core when its matrix does not fit the memory budget. The counts are then kept on
disk, and each step reads them a block of cells at a time. The steps, their arguments and their results
are the same as in memory.

## Sparse counts

<figure markdown="span">
  ![A 4 by 6 sparse matrix and its three arrays indptr, indices and data, with the values of cell 2 picked out in each](../assets/diagrams/csr-light.svg#only-light)
  ![A 4 by 6 sparse matrix and its three arrays indptr, indices and data, with the values of cell 2 picked out in each](../assets/diagrams/csr-dark.svg#only-dark)
  <figcaption>Compressed sparse rows: only the stored values are kept.</figcaption>
</figure>

A count matrix is mostly zeros, so it is stored as compressed sparse rows: for each cell, the genes it
expresses (`indices`) and their counts (`data`), with `indptr` marking where each cell starts. A
dataset's size is its number of stored values, not cells times genes.

## The copy on disk

<figure markdown="span">
  ![Rows grouped in chunks of 1,024; one chunk's gene ids and values split into byte planes; the three files of a store: indptr, the chunks back to back, and their offsets](../assets/diagrams/store-light.svg#only-light)
  ![Rows grouped in chunks of 1,024; one chunk's gene ids and values split into byte planes; the three files of a store: indptr, the chunks back to back, and their offsets](../assets/diagrams/store-dark.svg#only-dark)
  <figcaption>The copy of the counts in the project folder.</figcaption>
</figure>

When an analysis starts out of core, the counts are copied once into the project folder. Raw counts are
small whole numbers, so most of their bytes are zero; the copy groups the cells in chunks and stores
each chunk's low and high bytes apart, which compresses well: about 1.2 bytes per stored value on the
human fetal atlas, against 8 uncompressed. Reading gives back exactly the counts that were written, and
every chunk is checked as it is read, so a damaged file is an error and never a wrong count. The copy is
reused by later analyses of the same file.

## Filters and normalization as views

<figure markdown="span">
  ![The counts on disk with two genes and two cells left out by the filters, and a scale for each row applied as values are read](../assets/diagrams/views-light.svg#only-light)
  ![The counts on disk with two genes and two cells left out by the filters, and a scale for each row applied as values are read](../assets/diagrams/views-dark.svg#only-dark)
  <figcaption>Filters and normalization, recorded beside the counts.</figcaption>
</figure>

A filter records which cells or genes it keeps, and `normalize_total` and
`log1p` record a scale for each cell; every later step reads the counts through these records, and the copy
itself is never rewritten. Several analyses of the same cells (`adata.from_counts`) therefore share
one copy on disk.

See also: [Analyze a dataset larger than memory](../how-to/large-data.md),
[Memory and planning](memory.md).
