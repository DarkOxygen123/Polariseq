#![allow(
    clippy::doc_markdown,
    clippy::redundant_closure,
    clippy::unnecessary_wraps,
    clippy::case_sensitive_file_extension_comparisons
)]

//! 10x Genomics format readers.
//!
//! Two on-disk layouts:
//!
//! **`.h5` format** (Cell Ranger HDF5):
//! ```text
//! /matrix/data     — nnz × f32/f64 (column-major CSC values)
//! /matrix/indices  — nnz × i32/i64 (row indices per column)
//! /matrix/indptr   — (ncols+1) × i64 (column pointers)
//! /matrix/shape    — [nrows, ncols] = [n_genes, n_cells]
//! /matrix/barcodes — cell barcode strings
//! /matrix/features/name, /matrix/features/id — gene names/IDs
//! ```
//! Note: 10x stores genes × cells (transposed vs our cells × genes). We
//! transpose on load to match the AnnData convention.
//!
//! **MatrixMarket `.mtx` format** (three files in a directory):
//! ```text
//! matrix.mtx    — sparse matrix in MatrixMarket coordinate format
//! barcodes.tsv  — cell barcodes (one per line)
//! features.tsv  — gene info (id \t name \t type per line; older: genes.tsv)
//! ```

use hdf5_metno::File;
use polariseq_core::{CsrMatrix, Matrix};
use std::io::{BufRead, Read};
use std::path::Path;

use crate::{IoError, Result};

/// Read a 10x Genomics `.h5` file (Cell Ranger format).
///
/// Returns a cells × genes CSR matrix (transposed from 10x's genes × cells
/// storage) plus barcodes and gene names.
///
/// # Errors
/// Returns [`IoError`] on any read failure.
pub fn read_10x_h5(path: &str) -> Result<TenxData> {
    let file = File::open(path).map_err(IoError::Hdf5)?;
    let matrix_group = file
        .group("matrix")
        .map_err(|e| IoError::InvalidLayout(format!("no /matrix group: {e}")))?;

    // Read shape: [n_genes, n_cells]
    let shape_ds = matrix_group
        .dataset("shape")
        .map_err(|e| IoError::InvalidLayout(format!("no shape: {e}")))?;
    let shape: Vec<u64> = shape_ds
        .read_raw::<u32>()
        .map(|v| v.into_iter().map(u64::from).collect())
        .or_else(|_| shape_ds.read_raw::<u64>())?;
    if shape.len() != 2 {
        return Err(IoError::InvalidLayout(format!(
            "shape must be length 2, got {}",
            shape.len()
        )));
    }
    let n_genes = shape[0] as usize;
    let n_cells = shape[1] as usize;

    // Read the CSC triple (10x stores genes as rows, cells as columns).
    let data: Vec<f32> = read_flexible_f32(&matrix_group, "data")?;
    let indices: Vec<u32> = read_flexible_u32(&matrix_group, "indices")?;
    let indptr: Vec<u32> = read_flexible_u64_as_u32(&matrix_group, "indptr")?;

    if indptr.len() != n_cells + 1 {
        return Err(IoError::InvalidLayout(format!(
            "indptr length {} != n_cells+1 = {}",
            indptr.len(),
            n_cells + 1
        )));
    }

    // Transpose CSC (genes × cells) → CSR (cells × genes).
    let csr = csc_to_csr(&data, &indices, &indptr, n_genes, n_cells)?;

    // Read barcodes.
    let barcodes = read_string_array(&matrix_group, "barcodes")?;

    // Read gene names from features group.
    let features_group = matrix_group
        .group("features")
        .map_err(|e| IoError::InvalidLayout(format!("no features group: {e}")))?;
    let gene_names = read_string_array(&features_group, "name")?;
    let gene_ids = read_string_array(&features_group, "id")
        .ok()
        .unwrap_or_default();

    Ok(TenxData {
        matrix: Matrix::Csr(csr),
        barcodes,
        gene_names,
        gene_ids,
        n_cells,
        n_genes,
    })
}

/// The result of reading a 10x dataset.
pub struct TenxData {
    /// The cells × genes expression matrix.
    pub matrix: Matrix,
    /// Cell barcode strings.
    pub barcodes: Vec<String>,
    /// Gene display names.
    pub gene_names: Vec<String>,
    /// Gene Ensembl IDs.
    pub gene_ids: Vec<String>,
    /// Number of cells.
    pub n_cells: usize,
    /// Number of genes.
    pub n_genes: usize,
}

/// Read a 10x MatrixMarket directory (matrix.mtx + barcodes.tsv + features.tsv).
///
/// The `dir` should contain `matrix.mtx`, `barcodes.tsv`, and either
/// `features.tsv` (newer) or `genes.tsv` (older format). Handles both plain
/// and gzipped variants.
///
/// # Errors
/// Returns [`IoError`] on any read failure.
pub fn read_10x_mtx(dir: &str) -> Result<TenxData> {
    let dir_path = Path::new(dir);

    // Find the matrix file (plain or gzipped).
    let mtx_path = dir_path.join("matrix.mtx");
    let mtx_gz_path = dir_path.join("matrix.mtx.gz");
    let mtx_content: String = if mtx_path.exists() {
        std::fs::read_to_string(&mtx_path)
            .map_err(|e| IoError::InvalidLayout(format!("cannot read matrix.mtx: {e}")))?
    } else if mtx_gz_path.exists() {
        read_gzip_to_string(&mtx_gz_path)?
    } else {
        return Err(IoError::InvalidLayout(format!(
            "no matrix.mtx or matrix.mtx.gz in {dir}"
        )));
    };

    // Parse MatrixMarket coordinate format:
    // %%MatrixMarket matrix coordinate integer general
    // %metadata...
    // nrows ncols nnz
    // row col value
    let mut lines = mtx_content.lines().filter(|l| !l.trim().is_empty());
    // Skip header comments.
    let dims_line = loop {
        let line = lines
            .next()
            .ok_or_else(|| IoError::InvalidLayout("empty matrix.mtx".into()))?;
        if !line.starts_with('%') {
            break line;
        }
    };
    let dims: Vec<usize> = dims_line
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if dims.len() != 3 {
        return Err(IoError::InvalidLayout(format!(
            "bad matrix.mtx header: {dims_line}"
        )));
    }
    let (n_genes, n_cells, nnz) = (dims[0], dims[1], dims[2]);

    // Read all triplets.
    let mut rows: Vec<u32> = Vec::with_capacity(nnz);
    let mut cols: Vec<u32> = Vec::with_capacity(nnz);
    let mut vals: Vec<f32> = Vec::with_capacity(nnz);
    for (line_num, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            continue; // skip malformed
        }
        let row: u32 = parts[0]
            .parse()
            .map_err(|_| IoError::InvalidLayout(format!("bad row at line {}", line_num + 1)))?;
        let col: u32 = parts[1]
            .parse()
            .map_err(|_| IoError::InvalidLayout(format!("bad col at line {}", line_num + 1)))?;
        let val: f32 = parts[2]
            .parse()
            .map_err(|_| IoError::InvalidLayout(format!("bad val at line {}", line_num + 1)))?;
        rows.push(row - 1); // MatrixMarket is 1-indexed
        cols.push(col - 1);
        vals.push(val);
    }

    // Build CSC from triplets, then transpose to CSR.
    // (MatrixMarket stores genes as rows, cells as columns — same as .h5.)
    let csc = triplets_to_csc(&rows, &cols, &vals, n_genes, n_cells)?;
    let csr = csc_to_csr(&csc.0, &csc.1, &csc.2, n_genes, n_cells)?;

    // Read barcodes.
    let barcodes = read_tsv_lines(dir_path, &["barcodes.tsv", "barcodes.tsv.gz"])?;

    // Read features (newer) or genes (older).
    let (gene_names, gene_ids) =
        if dir_path.join("features.tsv").exists() || dir_path.join("features.tsv.gz").exists() {
            let lines = read_tsv_lines(dir_path, &["features.tsv", "features.tsv.gz"])?;
            let ids: Vec<String> = lines
                .iter()
                .map(|l| l.split('\t').next().unwrap_or("").to_string())
                .collect();
            let names: Vec<String> = lines
                .iter()
                .map(|l| l.split('\t').nth(1).unwrap_or("").to_string())
                .collect();
            (names, ids)
        } else if dir_path.join("genes.tsv").exists() || dir_path.join("genes.tsv.gz").exists() {
            let lines = read_tsv_lines(dir_path, &["genes.tsv", "genes.tsv.gz"])?;
            let ids: Vec<String> = lines
                .iter()
                .map(|l| l.split('\t').next().unwrap_or("").to_string())
                .collect();
            let names: Vec<String> = lines
                .iter()
                .map(|l| l.split('\t').nth(1).unwrap_or("").to_string())
                .collect();
            (names, ids)
        } else {
            (Vec::new(), Vec::new())
        };

    Ok(TenxData {
        matrix: Matrix::Csr(csr),
        barcodes,
        gene_names,
        gene_ids,
        n_cells,
        n_genes,
    })
}

// --- helpers -----------------------------------------------------------------

/// Read a dataset that could be f32 or f64.
fn read_flexible_f32(group: &hdf5_metno::Group, name: &str) -> Result<Vec<f32>> {
    let ds = group
        .dataset(name)
        .map_err(|e| IoError::InvalidLayout(format!("no {name}: {e}")))?;
    ds.read_raw::<f32>()
        .or_else(|_| {
            let v = ds.read_raw::<f64>()?;
            Ok(v.into_iter().map(|x| x as f32).collect::<Vec<f32>>())
        })
        .map_err(|e: hdf5_metno::Error| IoError::Hdf5(e))
}

/// Read a dataset that could be i32 or i64, as u32.
///
/// 10x writes `indptr` as int32, but a count matrix past 2^31 nonzeros needs
/// int64 — the 1.3M-cell mouse-brain file holds 2,624,828,308 — so the
/// stored dtype decides the memory type. See
/// [`crate::h5ad::widen_integer_dataset`] for why the int32-first conversion
/// path this replaced was wrong.
fn read_flexible_u32(group: &hdf5_metno::Group, name: &str) -> Result<Vec<u32>> {
    let ds = group
        .dataset(name)
        .map_err(|e| IoError::InvalidLayout(format!("no {name}: {e}")))?;
    crate::h5ad::widen_integer_dataset(&ds, name)
}

/// Read a dataset that could be i32/i64, as u32 (for indptr).
fn read_flexible_u64_as_u32(group: &hdf5_metno::Group, name: &str) -> Result<Vec<u32>> {
    read_flexible_u32(group, name)
}

/// Read a string array dataset. Handles variable-length and fixed-length
/// strings by dispatching on the HDF5 type descriptor.
fn read_string_array(group: &hdf5_metno::Group, name: &str) -> Result<Vec<String>> {
    let ds = group
        .dataset(name)
        .map_err(|e| IoError::InvalidLayout(format!("no {name}: {e}")))?;

    // Try variable-length types first (most common in newer files).
    if let Ok(v) = ds.read_raw::<hdf5_metno::types::VarLenUnicode>() {
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }
    if let Ok(v) = ds.read_raw::<hdf5_metno::types::VarLenAscii>() {
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }

    // For fixed-length strings, dispatch on the byte size to the right
    // const-generic FixedAscii<N> type. 10x files typically use 16 or 18.
    let dtype = ds.dtype().map_err(IoError::Hdf5)?;
    let size = dtype.size();
    macro_rules! try_fixed {
        ($n:expr) => {
            if size == $n {
                if let Ok(v) = ds.read_raw::<hdf5_metno::types::FixedAscii<$n>>() {
                    return Ok(v.into_iter().map(|s| s.to_string()).collect());
                }
            }
        };
    }
    try_fixed!(1);
    try_fixed!(2);
    try_fixed!(4);
    try_fixed!(8);
    try_fixed!(12);
    try_fixed!(14);
    try_fixed!(16);
    try_fixed!(18);
    try_fixed!(20);
    try_fixed!(24);
    try_fixed!(28);
    try_fixed!(32);
    try_fixed!(40);
    try_fixed!(48);
    try_fixed!(56);
    try_fixed!(64);
    try_fixed!(96);
    try_fixed!(128);
    try_fixed!(192);
    try_fixed!(256);

    Err(IoError::InvalidLayout(format!(
        "cannot read {name}: fixed-length string of {size} bytes not in dispatch table"
    )))
}

/// Transpose a CSC matrix (genes × cells) to CSR (cells × genes).
///
/// Mathematical insight: the CSC of a genes×cells matrix has EXACTLY the same
/// memory layout as the CSR of its transpose (cells×genes). The indptr indexes
/// columns (= cells) in CSC and rows (= cells) in CSR identically; the indices
/// are gene indices in both. So this is a reinterpretation, not a data movement.
fn csc_to_csr(
    data: &[f32],
    indices: &[u32],
    indptr: &[u32],
    n_genes: usize,
    n_cells: usize,
) -> Result<CsrMatrix> {
    // Verify indptr has the right length (one entry per cell + sentinel).
    if indptr.len() != n_cells + 1 {
        return Err(IoError::InvalidLayout(format!(
            "indptr len {} != n_cells+1 {}",
            indptr.len(),
            n_cells + 1
        )));
    }

    // Verify indices are in range [0, n_genes).
    for &idx in indices {
        if idx as usize >= n_genes {
            return Err(IoError::InvalidLayout(format!(
                "gene index {idx} out of range [0, {n_genes})"
            )));
        }
    }

    // The arrays are already in CSR (cells × genes) form — just reinterpret.
    CsrMatrix::new(
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
        n_cells,
        n_genes,
    )
    .map_err(IoError::from)
}

/// Convert coordinate triplets to CSC format.
fn triplets_to_csc(
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Result<(Vec<f32>, Vec<u32>, Vec<u32>)> {
    // Count per column.
    let mut col_counts = vec![0u32; n_cols];
    for &c in cols {
        col_counts[c as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(n_cols + 1);
    indptr.push(0u32);
    let mut running = 0u32;
    for &count in &col_counts {
        running += count;
        indptr.push(running);
    }

    let nnz = vals.len();
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut fill = indptr[..n_cols].to_vec();

    for k in 0..nnz {
        let col = cols[k] as usize;
        let pos = fill[col] as usize;
        indices[pos] = rows[k];
        data[pos] = vals[k];
        fill[col] += 1;
    }

    let _ = n_rows;
    Ok((data, indices, indptr))
}

/// Read a gzipped file to a string.
fn read_gzip_to_string(path: &Path) -> Result<String> {
    let f = std::fs::File::open(path)
        .map_err(|e| IoError::InvalidLayout(format!("cannot open {}: {e}", path.display())))?;
    let mut decoder = flate2::read::GzDecoder::new(f);
    let mut s = String::new();
    decoder
        .read_to_string(&mut s)
        .map_err(|e| IoError::InvalidLayout(format!("gzip decompress failed: {e}")))?;
    Ok(s)
}

/// Read lines from a TSV file (first column only).
fn read_tsv_lines(dir: &Path, names: &[&str]) -> Result<Vec<String>> {
    for name in names {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        if name.ends_with(".gz") {
            let content = read_gzip_to_string(&path)?;
            return Ok(content.lines().map(|l| l.trim().to_string()).collect());
        }
        let f = std::fs::File::open(&path)
            .map_err(|e| IoError::InvalidLayout(format!("cannot open {name}: {e}")))?;
        let reader = std::io::BufReader::new(f);
        return Ok(reader
            .lines()
            .map_while(std::result::Result::ok)
            .map(|l| l.trim().to_string())
            .collect());
    }
    Err(IoError::InvalidLayout(format!(
        "none of {names:?} found in {}",
        dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_to_csr_transpose() {
        // CSC (2 genes × 3 cells), stored column-by-column (columns = cells):
        //   cell 0: gene 0 = 1.0
        //   cell 1: gene 1 = 2.0
        //   cell 2: gene 0 = 3.0
        // CSC arrays: indptr = [0, 1, 2, 3] (one per cell + sentinel),
        //             indices = gene indices [0, 1, 0], data = [1.0, 2.0, 3.0]
        // The CSR of the transpose (3 cells × 2 genes) has identical layout.
        let data = vec![1.0_f32, 2.0, 3.0];
        let indices = vec![0_u32, 1, 0]; // gene indices
        let indptr = vec![0_u32, 1, 2, 3]; // cell boundaries

        let csr = csc_to_csr(&data, &indices, &indptr, 2, 3).unwrap();
        assert_eq!(csr.shape(), (3, 2)); // 3 cells × 2 genes
                                         // Cell 0: [1.0, 0.0] → gene 0 only
        let (cols, vals) = csr.row(0);
        assert_eq!(cols, &[0]);
        assert_eq!(vals, &[1.0]);
        // Cell 1: [0.0, 2.0] → gene 1 only
        let (cols, vals) = csr.row(1);
        assert_eq!(cols, &[1]);
        assert_eq!(vals, &[2.0]);
        // Cell 2: [3.0, 0.0] → gene 0 only
        let (cols, vals) = csr.row(2);
        assert_eq!(cols, &[0]);
        assert_eq!(vals, &[3.0]);
    }

    #[test]
    fn read_10x_h5_pbmc() {
        let path = "data/pbmc_1k_v2.h5";
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not downloaded");
            return;
        }
        let result = read_10x_h5(path).unwrap();
        assert!(
            result.n_cells > 500,
            "expected >500 cells, got {}",
            result.n_cells
        );
        assert!(
            result.n_genes > 10_000,
            "expected >10K genes, got {}",
            result.n_genes
        );
        assert_eq!(result.barcodes.len(), result.n_cells);
        assert_eq!(result.gene_names.len(), result.n_genes);
        println!(
            "PBMC 1k: {} cells × {} genes, nnz={}",
            result.n_cells,
            result.n_genes,
            match &result.matrix {
                Matrix::Csr(c) => c.nnz(),
                Matrix::Dense(_) => 0,
            }
        );
    }

    #[test]
    fn read_10x_mtx_pbmc3k() {
        let dir = "data/filtered_gene_bc_matrices/hg19";
        if !Path::new(dir).join("matrix.mtx").exists() {
            eprintln!("skipping: {dir} not found");
            return;
        }
        let result = read_10x_mtx(dir).unwrap();
        assert!(
            result.n_cells > 2000,
            "expected >2000 cells, got {}",
            result.n_cells
        );
        assert!(
            result.n_genes > 27_000,
            "expected >27K genes, got {}",
            result.n_genes
        );
        println!(
            "PBMC 3k mtx: {} cells × {} genes",
            result.n_cells, result.n_genes
        );
    }
}
