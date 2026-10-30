# Use Polariseq from Rust

The analysis runs in Rust, and its crates can be used directly from Rust code: the same algorithms
on the same sparse matrices, without Python.

<figure markdown="span">
  ![Four layers: the Python package, the bindings, polariseq-io beside polariseq-hdf5-raw, and polariseq-core at the bottom](../assets/diagrams/crates-light.svg#only-light)
  ![Four layers: the Python package, the bindings, polariseq-io beside polariseq-hdf5-raw, and polariseq-core at the bottom](../assets/diagrams/crates-dark.svg#only-dark)
  <figcaption>The crates. Each one uses those drawn below it.</figcaption>
</figure>

| crate | what it holds | reference |
|---|---|---|
| `polariseq-core` | the algorithms: quality control, normalization, PCA, the neighbor graph, Leiden, UMAP, Harmony, doublets, marker genes, cell type scores, pseudobulk | <a href="api/polariseq_core/">polariseq_core</a> |
| `polariseq-io` | reading `.h5ad` and 10x Genomics files, several files as one dataset, matrices larger than memory | <a href="api/polariseq_io/">polariseq_io</a> |

## Internal crates

The workspace holds two more crates, which are not meant to be used directly:

- `polariseq-hdf5-raw` holds the four low-level HDF5 calls that `polariseq-io` uses to read compressed
  chunks, so that `polariseq-io` itself contains no unsafe code. Read files through `polariseq-io`.
- `polariseq-python` builds `polariseq._polariseq`, the extension module the Python package imports, on
  top of `polariseq-core` and `polariseq-io`. It compiles to a Python extension, not a Rust library; its
  interface is the Python package, described in the [API reference](../api/index.md).

## Add the crates

The crates are not on crates.io yet. Depend on them from the repository, pinned to a commit:

```toml
[dependencies]
polariseq-core = { git = "https://github.com/DarkOxygen123/Polariseq", rev = "<commit>" }
polariseq-io = { git = "https://github.com/DarkOxygen123/Polariseq", rev = "<commit>" }
```

`polariseq-core` needs Rust 1.82 or later and no system libraries. `polariseq-io` builds HDF5 from
source on its first build, which takes a few minutes and needs a C compiler and CMake.

## A first program

Normalize a count matrix, then cluster its cells:

```rust
use polariseq_core::preprocess::{log1p_inplace, normalize_total_inplace};
use polariseq_core::{leiden, neighbors, pca_csr, CsrMatrix};
use polariseq_core::{LeidenOptions, NeighborsOptions, PcaOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 8 cells × 4 genes as compressed sparse rows.
    let indptr = vec![0, 2, 4, 6, 8, 10, 12, 14, 16];
    let indices = vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
    let data = vec![5.0, 3.0, 4.0, 2.0, 6.0, 3.0, 5.0, 4.0,
                    3.0, 6.0, 2.0, 5.0, 4.0, 4.0, 3.0, 5.0];
    let mut counts = CsrMatrix::new(indptr, indices, data, 8, 4)?;

    normalize_total_inplace(&mut counts, 1e4);
    log1p_inplace(&mut counts);

    let pcs = pca_csr(&counts, &PcaOptions { n_components: 2, ..Default::default() })?;
    let graph = neighbors(
        &pcs.embedding,
        pcs.n_obs,
        pcs.n_components,
        &NeighborsOptions { n_neighbors: 3, ..Default::default() },
    )?;
    let clusters = leiden(&graph, &LeidenOptions::default())?;
    println!("{} clusters: {:?}", clusters.n_communities, clusters.labels);
    Ok(())
}
```

To read a file instead, open it with `polariseq_io::AnnDataFile::open` and take its matrix with
`read_x`.

## Build the reference and run the tests

```bash
cargo doc --no-deps -p polariseq-core -p polariseq-io --open
```

```bash
cargo test -p polariseq-core -p polariseq-io
```

The crates are under the same license as the Python package: free for noncommercial use under the
PolyForm Noncommercial License 1.0.0 with additional terms, commercial use by agreement with the
authors.
