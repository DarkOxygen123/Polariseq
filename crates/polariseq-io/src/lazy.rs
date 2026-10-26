//! Lazy loading: metadata scanning and partial matrix reads.
//!
//! The key insight: HDF5 supports hyperslab selection — you can read a slice
//! of a dataset without loading the rest. For a CSR matrix stored as three
//! datasets (data, indices, indptr), reading rows [start, end) is:
//!
//! 1. Read `indptr[start..=end]` (end+1-start values — tiny)
//! 2. The nnz range is `[indptr[start], indptr[end])`
//! 3. Read `data[nnz_start..nnz_end]` and `indices[nnz_start..nnz_end]`
//!
//! This lets you peek at 100 cells from a 931K-cell dataset in milliseconds
//! instead of loading 8 GB into RAM.

use hdf5_metno::{Dataset, File};
use polariseq_core::CsrMatrix;
use std::path::Path;

use crate::{IoError, Result};

/// Lightweight dataset metadata — no matrix data read.
#[derive(Debug, Clone)]
pub struct DatasetMeta {
    /// File path.
    pub path: String,
    /// Number of cells (rows).
    pub n_cells: usize,
    /// Number of genes (columns).
    pub n_genes: usize,
    /// Number of stored nonzeros (0 if dense).
    pub nnz: usize,
    /// Whether X is sparse (CSR/CSC) or dense.
    pub is_sparse: bool,
    /// Cell/barcode names (may be empty if not present).
    pub cell_names: Vec<String>,
    /// Gene/feature names.
    pub gene_names: Vec<String>,
    /// Data dtype size in bytes (4 for f32, 8 for f64).
    pub dtype_bytes: usize,
    /// Estimated in-memory size if fully loaded (bytes).
    pub estimated_ram_bytes: usize,
    /// File size on disk (bytes).
    pub file_size_bytes: u64,
}

/// Scan an .h5ad file for metadata WITHOUT loading matrix data.
///
/// Reads only: shape attributes, name arrays, dataset shapes. This takes
/// milliseconds even for multi-GB files because HDF5 stores metadata in the
/// superblock — no data chunks are touched.
///
/// # Errors
/// Returns [`IoError`] if the file is not a valid h5ad.
pub fn scan_h5ad(path: &str) -> Result<DatasetMeta> {
    if !Path::new(path).exists() {
        return Err(IoError::InvalidLayout(format!("file not found: {path}")));
    }

    let file = File::open(path).map_err(IoError::Hdf5)?;
    let file_size = std::fs::metadata(path).map_or(0, |m| m.len());

    // Check /X group (sparse) or /X dataset (dense).
    let (n_cells, n_genes, nnz, is_sparse, dtype_bytes) = if let Ok(x_group) = file.group("X") {
        // Sparse: read shape attr + data dataset size.
        let shape: Vec<u64> = x_group
            .attr("shape")
            .map_err(|e| IoError::InvalidLayout(format!("no shape attr: {e}")))?
            .read_raw::<u64>()?;
        let nnz = x_group
            .dataset("data")
            .map_or(0, |d| d.shape().iter().product::<usize>());
        let dtype_size = x_group
            .dataset("data")
            .and_then(|d| d.dtype().map(|t| t.size()))
            .unwrap_or(4);
        (shape[0] as usize, shape[1] as usize, nnz, true, dtype_size)
    } else {
        // Dense: read the dataset shape directly.
        let ds = file
            .dataset("X")
            .map_err(|e| IoError::InvalidLayout(format!("no /X: {e}")))?;
        let shape = ds.shape();
        let dtype_size = ds.dtype().map_or(4, |t| t.size());
        (shape[0], shape[1], shape[0] * shape[1], false, dtype_size)
    };

    // Read names (small string arrays — fast).
    let cell_names = read_names(&file, "obs").unwrap_or_default();
    let gene_names = read_names(&file, "var").unwrap_or_default();

    // Estimate RAM: sparse = (nnz × 2 × dtype) + (n_cells+1) × 4; dense = n×m×dtype.
    let estimated = if is_sparse {
        nnz * dtype_bytes + nnz * 4 + (n_cells + 1) * 4
    } else {
        n_cells * n_genes * dtype_bytes
    };

    Ok(DatasetMeta {
        path: path.to_string(),
        n_cells,
        n_genes,
        nnz,
        is_sparse,
        cell_names,
        gene_names,
        dtype_bytes,
        estimated_ram_bytes: estimated,
        file_size_bytes: file_size,
    })
}

/// Read rows [start, end) from a sparse .h5ad file WITHOUT loading the rest.
///
/// Returns a CSR matrix containing only the requested rows. This is the
/// foundation for `head()`, `sample()`, and peek operations.
///
/// # Errors
/// Returns [`IoError`] on read failure or if the range is invalid.
pub fn read_rows_h5ad(path: &str, start: usize, end: usize) -> Result<CsrMatrix> {
    let file = File::open(path).map_err(IoError::Hdf5)?;
    let x = file
        .group("X")
        .map_err(|e| IoError::InvalidLayout(format!("X is not a sparse group: {e}")))?;

    // The fast path below slices `indptr` as *row* pointers, which is only
    // valid for CSR-encoded X. A CSC file's `indptr` indexes columns; reading
    // it as rows returns garbage that only surfaces downstream (found by the
    // telemetry harness: on a csc-encoded pbmc3k the spill arm's QC kept 256
    // of 2,700 cells). CSC row ranges are not contiguous on disk, so the
    // honest cost of correctness is one full read + counting-sort transpose
    // (O(nnz), never densified), then a row selection in memory.
    let encoding = crate::h5ad::read_attr_string(&x, "encoding-type")?;
    if encoding.as_deref() != Some("csr_matrix") {
        let g = &x;
        let shape: Vec<u64> = g
            .attr("shape")
            .map_err(|e| IoError::InvalidLayout(format!("no shape: {e}")))?
            .read_raw::<u64>()?;
        let (nrows, ncols) = (shape[0] as usize, shape[1] as usize);
        let data: Vec<f32> = g
            .dataset("data")
            .map_err(|e| IoError::InvalidLayout(format!("no data: {e}")))?
            .read_raw::<f32>()
            .map_err(IoError::Hdf5)?;
        let indices = crate::h5ad::widen_integer_dataset(
            &g.dataset("indices")
                .map_err(|e| IoError::InvalidLayout(format!("no indices: {e}")))?,
            "indices",
        )?;
        let indptr = crate::h5ad::widen_integer_dataset(
            &g.dataset("indptr")
                .map_err(|e| IoError::InvalidLayout(format!("no indptr: {e}")))?,
            "indptr",
        )?;
        let full =
            CsrMatrix::from_csc(&data, &indices, &indptr, nrows, ncols).map_err(IoError::from)?;
        let rows: Vec<usize> = (start..end).collect();
        return full.select_rows(&rows).map_err(IoError::from);
    }

    // Read shape.
    let shape: Vec<u64> = x
        .attr("shape")
        .map_err(|e| IoError::InvalidLayout(format!("no shape: {e}")))?
        .read_raw::<u64>()?;
    let (n_cells, n_genes) = (shape[0] as usize, shape[1] as usize);

    if start >= n_cells || end > n_cells || start >= end {
        return Err(IoError::InvalidLayout(format!(
            "invalid row range [{start}, {end}) for {n_cells} cells"
        )));
    }

    // Read the slice of indptr: [start, end] inclusive (end+1-start values).
    let indptr_ds = x
        .dataset("indptr")
        .map_err(|e| IoError::InvalidLayout(format!("no indptr: {e}")))?;
    // Row pointers read at full width: a file past 2^32 stored values has
    // 64-bit ones, and only the block's own offsets need to fit 32 bits.
    let indptr_slice: Vec<u64> = read_dataset_slice_u64(&indptr_ds, start, end + 1)?;

    // Compute the nnz range.
    let first = indptr_slice[0];
    let last = *indptr_slice
        .last()
        .ok_or_else(|| IoError::InvalidLayout("empty indptr slice".into()))?;
    let as_usize = |v: u64| {
        usize::try_from(v).map_err(|_| IoError::InvalidLayout(format!("row pointer {v} too large")))
    };
    let (nnz_start, nnz_end) = (as_usize(first)?, as_usize(last)?);

    // Rebase indptr to start from 0: offsets within the block.
    let rebased_indptr: Vec<u32> = indptr_slice
        .iter()
        .map(|&v| {
            u32::try_from(v - first).map_err(|_| {
                IoError::InvalidLayout(format!(
                    "rows {start}..{end} hold {} values, above the 2^32 a block can address",
                    last - first
                ))
            })
        })
        .collect::<Result<_>>()?;

    // Read data and indices for the nnz range.
    let data_ds = x
        .dataset("data")
        .map_err(|e| IoError::InvalidLayout(format!("no data: {e}")))?;
    let indices_ds = x
        .dataset("indices")
        .map_err(|e| IoError::InvalidLayout(format!("no indices: {e}")))?;

    let data: Vec<f32> = read_dataset_slice_f32(&data_ds, nnz_start, nnz_end)?;
    let indices: Vec<u32> = read_dataset_slice_u32(&indices_ds, nnz_start, nnz_end)?;

    CsrMatrix::new(rebased_indptr, indices, data, end - start, n_genes).map_err(IoError::from)
}

/// Detect the format of a dataset file from its extension and content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// `AnnData` HDF5 (.h5ad)
    H5ad,
    /// 10x Genomics HDF5 (.h5 with /matrix group)
    TenxH5,
    /// 10x `MatrixMarket` directory
    MtxDir,
}

/// Auto-detect the format of a dataset file.
///
/// # Errors
/// Returns [`IoError`] if the format cannot be determined.
pub fn detect_format(path: &str) -> Result<DataFormat> {
    let p = Path::new(path);
    if p.is_dir() {
        if p.join("matrix.mtx").exists() || p.join("matrix.mtx.gz").exists() {
            return Ok(DataFormat::MtxDir);
        }
        return Err(IoError::InvalidLayout(format!(
            "directory {path} has no matrix.mtx"
        )));
    }

    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "h5ad" => Ok(DataFormat::H5ad),
        "h5" => {
            // Check if it's an .h5ad (has /X group) or 10x (has /matrix group).
            let file = File::open(path).map_err(IoError::Hdf5)?;
            if file.group("matrix").is_ok() {
                Ok(DataFormat::TenxH5)
            } else if file.group("X").is_ok() {
                Ok(DataFormat::H5ad)
            } else {
                Err(IoError::InvalidLayout(format!(
                    "HDF5 file {path} has neither /matrix nor /X group"
                )))
            }
        }
        _ => Err(IoError::InvalidLayout(format!(
            "cannot detect format for {path} (extension: {ext})"
        ))),
    }
}

/// Scan metadata from any supported format (.h5ad, 10x .h5).
///
/// # Errors
/// Returns [`IoError`] if scanning fails.
pub fn scan_any(path: &str) -> Result<DatasetMeta> {
    match detect_format(path)? {
        DataFormat::H5ad => scan_h5ad(path),
        DataFormat::TenxH5 => scan_10x(path),
        DataFormat::MtxDir => {
            // For .mtx dirs, read the header of matrix.mtx for dimensions.
            scan_mtx_dir(path)
        }
    }
}

/// Scan a 10x .h5 file for metadata (no matrix data read).
fn scan_10x(path: &str) -> Result<DatasetMeta> {
    let file = File::open(path).map_err(IoError::Hdf5)?;
    let matrix = file
        .group("matrix")
        .map_err(|e| IoError::InvalidLayout(format!("no /matrix: {e}")))?;

    // Read shape: [n_genes, n_cells].
    let shape: Vec<u64> = matrix
        .dataset("shape")
        .map_err(|e| IoError::InvalidLayout(format!("no shape: {e}")))?
        .read_raw::<u32>()
        .map(|v| v.into_iter().map(u64::from).collect())
        .or_else(|_| matrix.dataset("shape")?.read_raw::<u64>())?;
    let (n_genes, n_cells) = (shape[0] as usize, shape[1] as usize);

    let nnz = matrix
        .dataset("data")
        .map_or(0, |d| d.shape().iter().product::<usize>());

    // Read barcodes from /matrix/barcodes.
    let cell_names: Vec<String> = read_10x_string_array(&matrix, "barcodes").unwrap_or_default();
    let gene_names = matrix
        .group("features")
        .ok()
        .and_then(|f| read_10x_string_array(&f, "name").ok())
        .unwrap_or_default();

    let file_size = std::fs::metadata(path).map_or(0, |m| m.len());
    let estimated = nnz * 8 + (n_cells + 1) * 4;

    Ok(DatasetMeta {
        path: path.to_string(),
        n_cells,
        n_genes,
        nnz,
        is_sparse: true,
        cell_names,
        gene_names,
        dtype_bytes: 4,
        estimated_ram_bytes: estimated,
        file_size_bytes: file_size,
    })
}

/// Read strings from a 10x HDF5 dataset (handles fixed and variable length).
fn read_10x_string_array(group: &hdf5_metno::Group, name: &str) -> Result<Vec<String>> {
    let ds = group
        .dataset(name)
        .map_err(|e| IoError::InvalidLayout(format!("no {name}: {e}")))?;
    if let Ok(v) = ds.read_raw::<hdf5_metno::types::VarLenUnicode>() {
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }
    if let Ok(v) = ds.read_raw::<hdf5_metno::types::VarLenAscii>() {
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }
    // Fixed-length fallback via byte reading.
    let dtype = ds.dtype().map_err(IoError::Hdf5)?;
    let size = dtype.size();
    if size > 0 && size <= 256 {
        // Try common sizes via macro.
        macro_rules! try_fix {
            ($n:expr) => {
                if size == $n {
                    if let Ok(v) = ds.read_raw::<hdf5_metno::types::FixedAscii<$n>>() {
                        return Ok(v.into_iter().map(|s| s.to_string()).collect());
                    }
                }
            };
        }
        try_fix!(16);
        try_fix!(18);
        try_fix!(32);
        try_fix!(64);
    }
    Err(IoError::InvalidLayout(format!(
        "cannot read {name} from 10x file"
    )))
}

/// Scan a .mtx directory for metadata.
fn scan_mtx_dir(dir: &str) -> Result<DatasetMeta> {
    let dir_path = Path::new(dir);
    let mtx_path = dir_path.join("matrix.mtx");
    let mtx_gz = dir_path.join("matrix.mtx.gz");

    let content: String = if mtx_path.exists() {
        std::fs::read_to_string(&mtx_path)
            .map_err(|e| IoError::InvalidLayout(format!("read matrix.mtx: {e}")))?
    } else if mtx_gz.exists() {
        let f = std::fs::File::open(&mtx_gz)
            .map_err(|e| IoError::InvalidLayout(format!("open matrix.mtx.gz: {e}")))?;
        let mut decoder = flate2::read::GzDecoder::new(f);
        let mut s = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut s)
            .map_err(|e| IoError::InvalidLayout(format!("decompress: {e}")))?;
        s
    } else {
        return Err(IoError::InvalidLayout("no matrix.mtx found".into()));
    };

    // Parse header: nrows ncols nnz.
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let dims_line = loop {
        let line = lines
            .next()
            .ok_or_else(|| IoError::InvalidLayout("empty matrix.mtx".into()))?;
        if !line.starts_with('%') {
            break line.to_string();
        }
    };
    // Count nnz from remaining lines.
    let nnz = lines.count();
    let dims: Vec<usize> = dims_line
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if dims.len() != 3 {
        return Err(IoError::InvalidLayout(format!(
            "bad mtx header: {dims_line}"
        )));
    }

    let (n_genes, n_cells, declared_nnz) = (dims[0], dims[1], dims[2]);

    // Read gene names from features.tsv or genes.tsv.
    let gene_names = read_mtx_gene_names(dir_path).unwrap_or_default();

    let file_size = std::fs::metadata(dir_path.join("matrix.mtx")).map_or(0, |m| m.len());

    Ok(DatasetMeta {
        path: dir.to_string(),
        n_cells,
        n_genes,
        nnz: declared_nnz.max(nnz),
        is_sparse: true,
        cell_names: Vec::new(),
        gene_names,
        dtype_bytes: 4,
        estimated_ram_bytes: declared_nnz * 8,
        file_size_bytes: file_size,
    })
}

fn read_mtx_gene_names(dir: &Path) -> std::io::Result<Vec<String>> {
    for name in [
        "features.tsv",
        "genes.tsv",
        "features.tsv.gz",
        "genes.tsv.gz",
    ] {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let is_gz = std::path::Path::new(name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gz"));
        let content = if is_gz {
            let f = std::fs::File::open(&path)?;
            let mut d = flate2::read::GzDecoder::new(f);
            let mut s = String::new();
            std::io::Read::read_to_string(&mut d, &mut s)?;
            s
        } else {
            std::fs::read_to_string(&path)?
        };
        return Ok(content
            .lines()
            .filter_map(|l| {
                let parts: Vec<&str> = l.split('\t').collect();
                parts.get(1).map(std::string::ToString::to_string) // gene name is 2nd column
            })
            .collect());
    }
    Ok(Vec::new())
}

/// Read rows [start, end) from any supported format.
///
/// # Errors
/// Returns [`IoError`] on read failure.
pub fn read_rows_any(path: &str, start: usize, end: usize) -> Result<CsrMatrix> {
    match detect_format(path)? {
        DataFormat::H5ad => read_rows_h5ad(path, start, end),
        DataFormat::TenxH5 => read_rows_10x(path, start, end),
        DataFormat::MtxDir => read_rows_mtx(path, start, end),
    }
}

/// Read rows from a 10x .h5 file.
///
/// The 10x format stores the matrix as CSC (genes × cells) with indptr
/// indexing CELLS. The insight from tenx.rs: CSC of genes×cells IS CSR of
/// cells×genes (identical memory layout), so we can read rows directly.
fn read_rows_10x(path: &str, start: usize, end: usize) -> Result<CsrMatrix> {
    let file = File::open(path).map_err(IoError::Hdf5)?;
    let matrix = file
        .group("matrix")
        .map_err(|e| IoError::InvalidLayout(format!("no /matrix: {e}")))?;

    // Read shape: [n_genes, n_cells].
    let shape: Vec<u64> = matrix
        .dataset("shape")
        .map_err(|e| IoError::InvalidLayout(format!("no shape: {e}")))?
        .read_raw::<u32>()
        .map(|v| v.into_iter().map(u64::from).collect())
        .or_else(|_| matrix.dataset("shape")?.read_raw::<u64>())?;
    let (n_genes, n_cells) = (shape[0] as usize, shape[1] as usize);

    if start >= n_cells || end > n_cells || start >= end {
        return Err(IoError::InvalidLayout(format!(
            "invalid row range [{start}, {end}) for {n_cells} cells"
        )));
    }

    // In 10x, indptr indexes cells (columns in CSC = rows in our CSR view).
    let indptr_ds = matrix
        .dataset("indptr")
        .map_err(|e| IoError::InvalidLayout(format!("no indptr: {e}")))?;
    let indptr_slice: Vec<u32> = read_dataset_slice_u32(&indptr_ds, start, end + 1)?;

    let nnz_start = indptr_slice[0] as usize;
    let nnz_end = *indptr_slice
        .last()
        .ok_or_else(|| IoError::InvalidLayout("empty".into()))? as usize;

    let rebased: Vec<u32> = indptr_slice.iter().map(|&v| v - nnz_start as u32).collect();

    let data_ds = matrix
        .dataset("data")
        .map_err(|e| IoError::InvalidLayout(format!("no data: {e}")))?;
    let indices_ds = matrix
        .dataset("indices")
        .map_err(|e| IoError::InvalidLayout(format!("no indices: {e}")))?;

    let data: Vec<f32> = read_dataset_slice_f32(&data_ds, nnz_start, nnz_end)?;
    let indices: Vec<u32> = read_dataset_slice_u32(&indices_ds, nnz_start, nnz_end)?;

    CsrMatrix::new(rebased, indices, data, end - start, n_genes).map_err(IoError::from)
}

/// Read rows from a .mtx directory (loads full matrix, extracts rows).
/// .mtx files are text — no partial read possible. For small files only.
fn read_rows_mtx(dir: &str, start: usize, end: usize) -> Result<CsrMatrix> {
    // Use the full 10x mtx reader, then slice.
    let tenx_data = crate::tenx::read_10x_mtx(dir)?;
    let polariseq_core::Matrix::Csr(csr) = tenx_data.matrix else {
        return Err(IoError::InvalidLayout("expected sparse".into()));
    };
    if start >= csr.nrows || end > csr.nrows || start >= end {
        return Err(IoError::InvalidLayout(format!(
            "range [{start}, {end}) invalid for {} rows",
            csr.nrows
        )));
    }
    // Slice the CSR.
    let indptr_slice = csr.indptr[start..=end].to_vec();
    let nnz_start = indptr_slice[0] as usize;
    let nnz_end = *indptr_slice.last().unwrap() as usize;
    let rebased: Vec<u32> = indptr_slice.iter().map(|&v| v - nnz_start as u32).collect();
    CsrMatrix::new(
        rebased,
        csr.indices[nnz_start..nnz_end].to_vec(),
        csr.data[nnz_start..nnz_end].to_vec(),
        end - start,
        csr.ncols,
    )
    .map_err(IoError::from)
}

/// Read a specific column (gene) from a sparse .h5ad file.
/// Returns (`cell_indices`, values) — only the cells expressing this gene.
///
/// # Errors
/// Returns [`IoError`] on read failure.
pub fn read_column_h5ad(path: &str, gene_index: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    // For column reads we need to scan all indices, but only load data for matches.
    let file = File::open(path).map_err(IoError::Hdf5)?;
    let x = file
        .group("X")
        .map_err(|e| IoError::InvalidLayout(format!("X is not sparse: {e}")))?;

    let all_indices = crate::h5ad::widen_integer_dataset(
        &x.dataset("indices")
            .map_err(|e| IoError::InvalidLayout(format!("no indices: {e}")))?,
        "indices",
    )?;

    let indptr = crate::h5ad::widen_integer_dataset(
        &x.dataset("indptr")
            .map_err(|e| IoError::InvalidLayout(format!("no indptr: {e}")))?,
        "indptr",
    )?;

    let n_cells = indptr.len() - 1;

    // Find matching entries.
    let mut cell_indices = Vec::new();
    let mut match_offsets = Vec::new();
    for cell in 0..n_cells {
        let (s, e) = (indptr[cell] as usize, indptr[cell + 1] as usize);
        // CSR holds at most one entry per (cell, gene).
        if let Some(k) = (s..e).find(|&k| all_indices[k] as usize == gene_index) {
            cell_indices.push(cell as u32);
            match_offsets.push(k);
        }
    }

    // Read only the matching data values.
    let data_ds = x
        .dataset("data")
        .map_err(|e| IoError::InvalidLayout(format!("no data: {e}")))?;
    let values: Vec<f32> = match_offsets
        .iter()
        .map(|&offset| read_dataset_slice_f32(&data_ds, offset, offset + 1).map(|v| v[0]))
        .collect::<Result<Vec<f32>>>()?;

    Ok((cell_indices, values))
}

// --- helpers -----------------------------------------------------------------

/// Read names from an obs/var group.
fn read_names(file: &File, group_name: &str) -> Result<Vec<String>> {
    let group = file
        .group(group_name)
        .map_err(|e| IoError::InvalidLayout(format!("no {group_name}: {e}")))?;

    // Try the _index attribute to find the index column name.
    let index_name = group
        .attr("_index")
        .and_then(|a| {
            a.read_scalar::<hdf5_metno::types::VarLenUnicode>()
                .map(|s| s.to_string())
        })
        .or_else(|_| {
            group.attr("_index").and_then(|a| {
                let v: Vec<hdf5_metno::types::VarLenUnicode> = a.read_raw()?;
                Ok(v.into_iter()
                    .next()
                    .map(|s| s.to_string())
                    .unwrap_or_default())
            })
        })
        .unwrap_or_else(|_| "_index".to_string());

    // Try group/values (nullable-string-array) or direct dataset.
    if let Ok(idx_group) = group.group(&index_name) {
        if let Ok(values) = idx_group.dataset("values") {
            let v: Vec<hdf5_metno::types::VarLenUnicode> =
                values.read_raw().map_err(IoError::Hdf5)?;
            return Ok(v.into_iter().map(|s| s.to_string()).collect());
        }
    }
    if let Ok(ds) = group.dataset(&index_name) {
        let v: Vec<hdf5_metno::types::VarLenUnicode> = ds.read_raw().map_err(IoError::Hdf5)?;
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }
    Ok(Vec::new())
}

/// Read a contiguous slice from an integer dataset using hyperslab selection.
///
/// Dispatches on the stored dtype for the same reason
/// [`h5ad::widen_integer_dataset`] does: an int64 `indptr` read through an
/// int32 memory type is converted silently, and a row pointer past 2^31
/// nonzeros comes back wrong while every length check still passes.
/// `[start, end)` of an integer dataset (32- or 64-bit, signed or not) as u64:
/// row pointers, which pass 2^32 in files of more stored values.
fn read_dataset_slice_u64(ds: &Dataset, start: usize, end: usize) -> Result<Vec<u64>> {
    use hdf5_metno::types::{IntSize, TypeDescriptor};
    let count = end - start;
    let sel: hdf5_metno::Selection = (start..end).into();
    let wide = matches!(
        ds.dtype()
            .map_err(IoError::Hdf5)?
            .to_descriptor()
            .map_err(IoError::Hdf5)?,
        TypeDescriptor::Integer(IntSize::U8) | TypeDescriptor::Unsigned(IntSize::U8)
    );
    let out: Vec<u64> = if wide {
        let slice: Vec<i64> = ds
            .read_slice_1d(sel)
            .map(|a| a.to_vec())
            .map_err(IoError::Hdf5)?;
        slice
            .into_iter()
            .map(|x| {
                u64::try_from(x)
                    .map_err(|_| IoError::InvalidLayout(format!("negative row pointer {x}")))
            })
            .collect::<Result<_>>()?
    } else {
        let slice: Vec<i32> = ds
            .read_slice_1d(sel)
            .map(|a| a.to_vec())
            .map_err(IoError::Hdf5)?;
        slice
            .into_iter()
            .map(|x| {
                u64::try_from(x)
                    .map_err(|_| IoError::InvalidLayout(format!("negative row pointer {x}")))
            })
            .collect::<Result<_>>()?
    };
    if out.len() != count {
        return Err(IoError::InvalidLayout(format!(
            "slice read returned {} elements, expected {count}",
            out.len()
        )));
    }
    Ok(out)
}

fn read_dataset_slice_u32(ds: &Dataset, start: usize, end: usize) -> Result<Vec<u32>> {
    use hdf5_metno::types::{IntSize, TypeDescriptor};
    let count = end - start;
    let sel: hdf5_metno::Selection = (start..end).into();
    let is_64 = matches!(
        ds.dtype()
            .map_err(IoError::Hdf5)?
            .to_descriptor()
            .map_err(IoError::Hdf5)?,
        TypeDescriptor::Integer(IntSize::U8) | TypeDescriptor::Unsigned(IntSize::U8)
    );
    if is_64 {
        let slice: Vec<i64> = ds
            .read_slice_1d(sel)
            .map(|a| a.to_vec())
            .map_err(IoError::Hdf5)?;
        if slice.len() == count {
            return slice
                .into_iter()
                .map(|x| {
                    u32::try_from(x).map_err(|_| {
                        IoError::InvalidLayout(format!(
                            "slice holds {x}, above the u32 index limit {}",
                            u32::MAX
                        ))
                    })
                })
                .collect();
        }
    } else {
        let slice: Vec<i32> = ds
            .read_slice_1d(sel)
            .map(|a| a.to_vec())
            .map_err(IoError::Hdf5)?;
        if slice.len() == count {
            return Ok(slice.into_iter().map(|v| v as u32).collect());
        }
    }
    Err(IoError::InvalidLayout(format!(
        "slice read returned wrong element count, expected {count}"
    )))
}

/// Read a contiguous slice from an f32 dataset.
fn read_dataset_slice_f32(ds: &Dataset, start: usize, end: usize) -> Result<Vec<f32>> {
    let count = end - start;
    let sel: hdf5_metno::Selection = (start..end).into();
    let slice: Vec<f32> = ds
        .read_slice_1d(sel)
        .map(|a| a.to_vec())
        .map_err(IoError::Hdf5)?;
    if slice.len() == count {
        return Ok(slice);
    }
    Err(IoError::InvalidLayout(format!(
        "slice read returned {} elements, expected {count}",
        slice.len()
    )))
}

/// Validate a dataset file: check structure, shapes, basic sanity.
///
/// # Errors
/// Returns a descriptive error for each problem found.
pub fn validate_h5ad(path: &str) -> Result<ValidationReport> {
    let meta = scan_h5ad(path)?;

    let mut issues = Vec::new();

    // Check shape sanity.
    if meta.n_cells == 0 {
        issues.push("0 cells".to_string());
    }
    if meta.n_genes == 0 {
        issues.push("0 genes".to_string());
    }
    if meta.nnz > meta.n_cells.saturating_mul(meta.n_genes) {
        issues.push(format!(
            "nnz ({}) > n_cells × n_genes ({})",
            meta.nnz,
            meta.n_cells * meta.n_genes
        ));
    }
    if meta.is_sparse && meta.nnz == 0 {
        issues.push("sparse matrix with 0 nonzeros".to_string());
    }
    // Check density (as percentage).
    let density = (meta.nnz as f64 / (meta.n_cells * meta.n_genes).max(1) as f64) * 100.0;
    if density > 90.0 {
        issues.push(format!(
            "high density ({density}%) for sparse — consider dense storage"
        ));
    }

    Ok(ValidationReport {
        valid: issues.is_empty(),
        issues,
        meta,
    })
}

/// Validation result.
#[derive(Debug)]
pub struct ValidationReport {
    /// Whether the dataset passed all checks.
    pub valid: bool,
    /// List of issues found (empty if valid).
    pub issues: Vec<String>,
    /// The scanned metadata.
    pub meta: DatasetMeta,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_read_int64_indptr_above_i32_max() {
        // The spill path reads indptr in per-block hyperslab slices; the
        // slice reader must follow the stored dtype for the same reason the
        // whole-dataset reader does.
        let mut path = std::env::temp_dir();
        path.push(format!("polariseq_slice_i64_{}.h5", std::process::id()));
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let ds = file.new_dataset::<i64>().shape(3).create("indptr").unwrap();
            ds.write_raw(&[0i64, 2_147_483_648, 2_624_828_308]).unwrap();
        }
        let file = File::open(&path).unwrap();
        let ds = file.dataset("indptr").unwrap();
        let v = read_dataset_slice_u32(&ds, 1, 3).unwrap();
        assert_eq!(v, vec![2_147_483_648u32, 2_624_828_308]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn scan_h5ad_metadata_only() {
        let path = "../../tests/data/sparse_100x200.h5ad";
        let meta = scan_h5ad(path).unwrap();
        assert_eq!(meta.n_cells, 100);
        assert_eq!(meta.n_genes, 200);
        assert_eq!(meta.nnz, 1719);
        assert!(meta.is_sparse);
        assert_eq!(meta.cell_names.len(), 100);
        assert_eq!(meta.gene_names.len(), 200);
    }

    #[test]
    fn read_rows_partial() {
        let path = "../../tests/data/sparse_100x200.h5ad";
        let csr = read_rows_h5ad(path, 10, 20).unwrap();
        assert_eq!(csr.nrows, 10);
        assert_eq!(csr.ncols, 200);
        // Should have some nonzeros.
        assert!(csr.nnz() > 0);
        // indptr should start at 0 (rebased).
        assert_eq!(csr.indptr[0], 0);
    }

    #[test]
    fn read_rows_first_and_last() {
        let path = "../../tests/data/sparse_100x200.h5ad";
        let first = read_rows_h5ad(path, 0, 5).unwrap();
        assert_eq!(first.nrows, 5);
        let last = read_rows_h5ad(path, 95, 100).unwrap();
        assert_eq!(last.nrows, 5);
    }

    #[test]
    fn read_rows_invalid_range() {
        let path = "../../tests/data/sparse_100x200.h5ad";
        assert!(read_rows_h5ad(path, 50, 50).is_err()); // start == end
        assert!(read_rows_h5ad(path, 99, 101).is_err()); // end > n_cells
    }

    #[test]
    fn validate_valid_dataset() {
        let path = "../../tests/data/sparse_100x200.h5ad";
        let report = validate_h5ad(path).unwrap();
        assert!(report.valid, "issues: {:?}", report.issues);
        assert_eq!(report.meta.n_cells, 100);
    }

    #[test]
    fn scan_10x_h5_if_present() {
        let path = "../../data/pbmc_1k_v2.h5";
        if !Path::new(path).exists() {
            return; // skip if not downloaded
        }
        // 10x .h5 uses a different layout — scan_h5ad only works for .h5ad.
        // This test verifies we don't crash on it.
        let _ = scan_h5ad(path);
    }
}
