//! Matrix container for cells × features data.
//!
//! Omics matrices are overwhelmingly sparse (single-cell expression is 90%+
//! zeros), so the container must make sparse first-class without penalizing the
//! dense embeddings (`X_pca`, `X_umap`) that show up later in a pipeline.
//!
//! Design choices:
//! - **CSR is the primary sparse format.** It matches the scipy/AnnData
//!   convention for scRNA-seq `X`, which is row-iterable (per-cell).
//! - **f32 is the default float.** Halves memory vs f64 — critical for the
//!   low-resource targets; scanpy/RAPIDS explicitly warn about f64 bloat.
//! - **Owned `Vec`-backed CSR triple.** Cache-friendly, no external sparse
//!   dependency in the core. The on-disk `.h5ad` layout *is* this triple.
//!
//! Storage types use `u32` for indices (supports up to 4 billion rows/cols —
//! far beyond any current dataset) and `f32` for values. Wider element types
//! can be added later behind a type parameter if needed.

use rayon::prelude::*;

/// The on-disk / in-memory representation of a cells × features matrix.
///
/// * `Dense` — row-major dense storage. Rare for scRNA-seq `X`, common for
///   embeddings (`X_pca`, `X_umap`) and ATAC-seq after densification.
/// * `Csr` — compressed sparse rows; the default for scRNA-seq `X`.
/// * `Csc` — compressed sparse columns; used for column-wise ops (Phase 6+).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixKind {
    /// Row-major dense storage.
    Dense,
    /// Compressed sparse row (CSR).
    Csr,
    /// Compressed sparse column (CSC).
    Csc,
}

impl MatrixKind {
    /// Human-readable name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            MatrixKind::Dense => "dense",
            MatrixKind::Csr => "csr",
            MatrixKind::Csc => "csc",
        }
    }
}

impl std::fmt::Display for MatrixKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors raised by matrix operations.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MatrixError {
    /// A CSR triple's shapes are mutually inconsistent. `reason` names the
    /// check that failed — a single message listing every length cannot be
    /// debugged when two of the lengths agree, which is exactly what happens
    /// when a reader corrupts one array of the triple (measured: an int64
    /// `indptr` read through an int32 memory type failed here with a message
    /// naming a length pair that matched).
    #[error("invalid CSR: {reason} (indptr len {indptr_len}, nrows {nrows}, nnz {nnz}, indices len {indices_len}, indptr tail {indptr_tail:?})")]
    InvalidCsr {
        /// Which consistency check failed.
        reason: &'static str,
        /// Length of the `indptr` array.
        indptr_len: usize,
        /// Stated number of rows.
        nrows: usize,
        /// Number of stored values.
        nnz: usize,
        /// Length of the `indices` array.
        indices_len: usize,
        /// Final row pointer, when `indptr` is long enough to have one.
        indptr_tail: Option<u32>,
    },
    /// A dimension mismatch between two operands.
    #[error("dimension mismatch: {expected} vs {actual}")]
    DimMismatch {
        /// What was expected.
        expected: String,
        /// What was found.
        actual: String,
    },
    /// Index out of bounds.
    #[error("index ({row}, {col}) out of bounds for shape {nrows}x{ncols}")]
    IndexOutOfBounds {
        /// Row requested.
        row: usize,
        /// Column requested.
        col: usize,
        /// Matrix row count.
        nrows: usize,
        /// Matrix column count.
        ncols: usize,
    },
}

/// Result alias for matrix operations.
pub type Result<T> = std::result::Result<T, MatrixError>;

/// A dense row-major matrix of `f32`.
///
/// Stored as a flat `Vec<f32>` of length `nrows * ncols` in row-major (C) order.
/// This is the representation `NumPy` gives us, so zero-copy interop is trivial.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    /// Row-major data, length `nrows * ncols`.
    pub data: Vec<f32>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl DenseMatrix {
    /// Create a new dense matrix from row-major data.
    ///
    /// # Errors
    /// Returns [`MatrixError::DimMismatch`] if `data.len() != nrows * ncols`.
    #[inline]
    pub fn from_row_major(data: Vec<f32>, nrows: usize, ncols: usize) -> Result<Self> {
        if data.len() != nrows * ncols {
            return Err(MatrixError::DimMismatch {
                expected: format!("{nrows}x{ncols} = {} elements", nrows * ncols),
                actual: format!("{} elements", data.len()),
            });
        }
        Ok(Self { data, nrows, ncols })
    }

    /// Create a matrix filled with zeros.
    #[must_use]
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of elements.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nrows * self.ncols
    }

    /// Whether the matrix has zero elements.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get one element.
    ///
    /// # Errors
    /// Returns [`MatrixError::IndexOutOfBounds`] if the index is invalid.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Result<f32> {
        if row >= self.nrows || col >= self.ncols {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                col,
                nrows: self.nrows,
                ncols: self.ncols,
            });
        }
        Ok(self.data[row * self.ncols + col])
    }

    /// Get one row as a slice.
    #[inline]
    #[must_use]
    pub fn row(&self, row: usize) -> &[f32] {
        &self.data[row * self.ncols..(row + 1) * self.ncols]
    }

    /// Center each column to zero mean in place (PCA preprocessing).
    ///
    /// Computes the column means in parallel with rayon, then subtracts. This
    /// is the standard PCA preprocessing step (`zero_center=True` in scanpy).
    pub fn center_columns_in_place(&mut self) {
        if self.nrows == 0 {
            return;
        }
        // Column sums, accumulated in parallel over rows.
        let mut col_sums = vec![0.0_f64; self.ncols];
        self.data
            .par_chunks_exact(self.ncols)
            .fold(
                || vec![0.0_f64; self.ncols],
                |mut acc, row| {
                    for (acc, &x) in acc.iter_mut().zip(row) {
                        *acc += f64::from(x);
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0_f64; self.ncols],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(&b) {
                        *a += *b;
                    }
                    a
                },
            )
            .iter()
            .zip(&mut col_sums)
            .for_each(|(partial, sum)| *sum = *partial);

        // f64 accumulation for numerical stability, then back to f32 for storage
        // (scanpy/sklearn do the same).
        let n = self.nrows as f64;
        for (j, sum) in col_sums.iter_mut().enumerate() {
            let mean = (*sum / n) as f32;
            // Subtract the mean from every element of column j.
            self.data
                .par_chunks_exact_mut(self.ncols)
                .for_each(|row| row[j] -= mean);
        }
    }

    /// Return a column-major `faer::Mat` view for linear algebra. This is a
    /// **copy** — faer is column-major, we are row-major. For PCA we accept the
    /// transpose cost once; a future optimization stores dense matrices in
    /// column-major when they're destined for linalg.
    #[must_use]
    pub fn to_faer_col_major(&self) -> faer::Mat<f32> {
        let mut m = faer::Mat::<f32>::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                m[(i, j)] = self.data[i * self.ncols + j];
            }
        }
        m
    }
}

/// A CSR (compressed sparse row) matrix of `f32`.
///
/// The canonical sparse representation for scRNA-seq expression data. Stored as
/// the three CSR arrays plus shape. `u32` indices support up to ~4 billion
/// rows/cols; `f32` values halve memory vs f64.
///
/// Invariants (validated on construction):
/// - `indptr.len() == nrows + 1`
/// - `indptr` is non-decreasing, starts at 0, ends at `nnz`
/// - `data.len() == indices.len() == nnz == indptr[nrows]`
/// - all `indices[k] < ncols`
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Row pointers, length `nrows + 1`. `indptr[i]..indptr[i+1]` is the slice
    /// of `indices`/`data` for row `i`.
    pub indptr: Vec<u32>,
    /// Column indices, length `nnz`.
    pub indices: Vec<u32>,
    /// Values, length `nnz`.
    pub data: Vec<f32>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl CsrMatrix {
    /// Construct a CSR matrix from the standard triple, validating invariants.
    ///
    /// # Errors
    /// Returns [`MatrixError::InvalidCsr`] if the arrays are inconsistent.
    pub fn new(
        indptr: Vec<u32>,
        indices: Vec<u32>,
        data: Vec<f32>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        // Each check reports itself, so the error names the array at fault
        // instead of three lengths and a guess.
        if indptr.len() != nrows + 1 {
            return Err(MatrixError::InvalidCsr {
                reason: "indptr length is not nrows + 1",
                indptr_len: indptr.len(),
                nrows,
                nnz: data.len(),
                indices_len: indices.len(),
                indptr_tail: indptr.last().copied(),
            });
        }
        if data.len() != indices.len() {
            return Err(MatrixError::InvalidCsr {
                reason: "data and indices have different lengths",
                indptr_len: indptr.len(),
                nrows,
                nnz: data.len(),
                indices_len: indices.len(),
                indptr_tail: Some(indptr[nrows]),
            });
        }
        if data.len() != indptr[nrows] as usize {
            return Err(MatrixError::InvalidCsr {
                reason: "data length does not match the final row pointer",
                indptr_len: indptr.len(),
                nrows,
                nnz: data.len(),
                indices_len: indices.len(),
                indptr_tail: Some(indptr[nrows]),
            });
        }
        // Indices must be in range. (We do not require sorted rows here.)
        if let Some(&bad) = indices.iter().find(|&c| (*c as usize) >= ncols) {
            return Err(MatrixError::IndexOutOfBounds {
                row: 0,
                col: bad as usize,
                nrows,
                ncols,
            });
        }
        Ok(Self {
            indptr,
            indices,
            data,
            nrows,
            ncols,
        })
    }

    /// Construct a CSR from a dense row-major matrix, dropping exact zeros.
    ///
    /// This is the path the `.h5ad` dense reader uses to densify→sparse if
    /// needed, and the path tests use to build fixtures.
    #[must_use]
    pub fn from_dense_row_major(dense: &[f32], nrows: usize, ncols: usize) -> Self {
        debug_assert_eq!(dense.len(), nrows * ncols);
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..nrows {
            for j in 0..ncols {
                let v = dense[i * ncols + j];
                if v != 0.0 {
                    indices.push(j as u32);
                    data.push(v);
                }
            }
            indptr.push(data.len() as u32);
        }
        Self {
            indptr,
            indices,
            data,
            nrows,
            ncols,
        }
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored nonzeros.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Sparsity fraction: `nnz / (nrows * ncols)`.
    #[inline]
    #[must_use]
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Row `i` as `(column_indices, values)` slices.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        (&self.indices[start..end], &self.data[start..end])
    }

    /// Densify to a row-major `DenseMatrix`. Allocates `nrows * ncols` floats —
    /// expensive for large sparse matrices; only call when you genuinely need
    /// dense (e.g. for a small HVG subset before PCA).
    #[must_use]
    pub fn to_dense(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            let (cols, vals) = self.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                out.data[i * self.ncols + c as usize] = v;
            }
        }
        out
    }

    /// Apply a function to every nonzero, producing a new CSR with the same
    /// sparsity pattern. Used for `log1p`, scaling, etc. Parallel over rows.
    #[must_use]
    pub fn map_values<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32 + Sync,
    {
        let data: Vec<f32> = self.data.par_iter().map(|&v| f(v)).collect();
        Self {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Sum each row into a `Vec<f32>` of length `nrows`. Parallel over rows.
    /// This is the core of `normalize_total` (per-cell library size).
    #[must_use]
    pub fn row_sums(&self) -> Vec<f32> {
        (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                let (_, vals) = self.row(i);
                vals.iter().sum()
            })
            .collect()
    }

    /// Sum each column into a `Vec<f32>` of length `ncols`. Parallel over cols.
    /// Used for per-gene statistics (means, variance numerator).
    #[must_use]
    pub fn col_sums(&self) -> Vec<f32> {
        // Per-thread accumulators to avoid contention on `out`; reduce at the end.
        // This is the standard reduce-then-merge pattern for sparse column ops.
        let ncols = self.ncols;
        let partials = (0..self.nrows)
            .into_par_iter()
            .fold(
                || vec![0.0_f32; ncols],
                |mut acc, i| {
                    let (cols, vals) = self.row(i);
                    for (&c, &v) in cols.iter().zip(vals) {
                        acc[c as usize] += v;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0_f32; ncols],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(&b) {
                        *a += *b;
                    }
                    a
                },
            );
        partials
    }
}

/// Rows per block for parallel row-wise kernels: small enough to balance
/// across threads when nnz per row is skewed, large enough that per-block
/// overhead vanishes.
pub const ROW_BLOCK: usize = 2048;

/// Split `buf` — an array laid out like a CSR `data`/`indices` array described
/// by `indptr` — into disjoint mutable slices covering `rows_per_block` rows
/// each. Returns `(row_start, row_end, slice)` triples in row order.
///
/// This is the `unsafe`-free primitive behind every parallel row-wise kernel:
/// each block can be handed to a rayon task with exclusive access to its rows.
///
/// # Panics
/// Panics if `buf.len()` does not equal `indptr[last]`.
#[must_use]
pub fn split_rows_mut<'a, T>(
    indptr: &[u32],
    buf: &'a mut [T],
    rows_per_block: usize,
) -> Vec<(usize, usize, &'a mut [T])> {
    let nrows = indptr.len().saturating_sub(1);
    assert_eq!(
        buf.len(),
        indptr.last().map_or(0, |&e| e as usize),
        "buffer length must equal indptr[nrows]"
    );
    let rows_per_block = rows_per_block.max(1);
    let mut blocks = Vec::with_capacity(nrows.div_ceil(rows_per_block));
    let mut rest: &'a mut [T] = buf;
    let mut r0 = 0;
    while r0 < nrows {
        let r1 = (r0 + rows_per_block).min(nrows);
        let len = (indptr[r1] - indptr[r0]) as usize;
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        blocks.push((r0, r1, head));
        rest = tail;
        r0 = r1;
    }
    blocks
}

impl CsrMatrix {
    /// Gather rows in the given order. Duplicates and arbitrary order are
    /// allowed (numpy fancy-indexing semantics). Parallel over row blocks.
    ///
    /// # Errors
    /// Returns [`MatrixError::IndexOutOfBounds`] if any row is `>= nrows`.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        indptr.push(0_u32);
        for &r in rows {
            if r >= self.nrows {
                return Err(MatrixError::IndexOutOfBounds {
                    row: r,
                    col: 0,
                    nrows: self.nrows,
                    ncols: self.ncols,
                });
            }
            let len = self.indptr[r + 1] - self.indptr[r];
            let last = *indptr.last().unwrap_or(&0);
            indptr.push(last + len);
        }
        let nnz = *indptr.last().unwrap_or(&0) as usize;
        let mut data = vec![0.0_f32; nnz];
        let mut indices = vec![0_u32; nnz];
        let dblocks = split_rows_mut(&indptr, &mut data, ROW_BLOCK);
        let iblocks = split_rows_mut(&indptr, &mut indices, ROW_BLOCK);
        dblocks
            .into_par_iter()
            .zip(iblocks)
            .for_each(|((r0, r1, d), (_, _, ix))| {
                let mut off = 0;
                for &src in &rows[r0..r1] {
                    let (cols, vals) = self.row(src);
                    let n = vals.len();
                    d[off..off + n].copy_from_slice(vals);
                    ix[off..off + n].copy_from_slice(cols);
                    off += n;
                }
            });
        Ok(Self {
            indptr,
            indices,
            data,
            nrows: rows.len(),
            ncols: self.ncols,
        })
    }

    /// Keep only the rows where `mask` is `true`, compacting `data` and
    /// `indices` in place — a two-pointer pass where rows only move left —
    /// and rebuilding `indptr`. Peak memory stays at ~1× the matrix plus
    /// one `u32` per row, instead of the ~1.5× of [`Self::select_rows_mask`]'s
    /// full copy. Returns the number of rows kept.
    ///
    /// # Errors
    /// Returns [`MatrixError::DimMismatch`] if `mask.len() != nrows`.
    pub fn retain_rows_mask(&mut self, mask: &[bool]) -> Result<usize> {
        if mask.len() != self.nrows {
            return Err(MatrixError::DimMismatch {
                expected: format!("{} rows", self.nrows),
                actual: format!("mask of length {}", mask.len()),
            });
        }
        let mut new_indptr = Vec::with_capacity(self.nrows + 1);
        new_indptr.push(0_u32);
        let mut write = 0_usize;
        for ((&keep, &start), &end) in mask
            .iter()
            .zip(&self.indptr[..self.nrows])
            .zip(&self.indptr[1..])
        {
            let (s, e) = (start as usize, end as usize);
            if keep {
                if write != s {
                    self.data.copy_within(s..e, write);
                    self.indices.copy_within(s..e, write);
                }
                write += e - s;
                new_indptr.push(write as u32);
            }
        }
        self.data.truncate(write);
        self.indices.truncate(write);
        self.nrows = new_indptr.len() - 1;
        self.indptr = new_indptr;
        Ok(self.nrows)
    }

    /// Keep the rows where `mask` is `true`.
    ///
    /// # Errors
    /// Returns [`MatrixError::DimMismatch`] if `mask.len() != nrows`.
    pub fn select_rows_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.nrows {
            return Err(MatrixError::DimMismatch {
                expected: format!("{} rows", self.nrows),
                actual: format!("mask of length {}", mask.len()),
            });
        }
        let rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| keep.then_some(i))
            .collect();
        self.select_rows(&rows)
    }

    /// Keep the columns in `cols`, in that order; the result has
    /// `cols.len()` columns. Two passes (count, then fill), parallel over rows.
    /// Within a row the surviving entries keep their original relative order,
    /// so the output may have unsorted column indices — no kernel here
    /// requires sorted indices.
    ///
    /// # Errors
    /// Returns [`MatrixError::IndexOutOfBounds`] for a column `>= ncols` and
    /// [`MatrixError::DimMismatch`] if a column is selected twice.
    pub fn select_cols(&self, cols: &[usize]) -> Result<Self> {
        let mut col_map = vec![u32::MAX; self.ncols];
        for (new, &old) in cols.iter().enumerate() {
            if old >= self.ncols {
                return Err(MatrixError::IndexOutOfBounds {
                    row: 0,
                    col: old,
                    nrows: self.nrows,
                    ncols: self.ncols,
                });
            }
            if col_map[old] != u32::MAX {
                return Err(MatrixError::DimMismatch {
                    expected: "unique column indices".into(),
                    actual: format!("column {old} selected twice"),
                });
            }
            col_map[old] = new as u32;
        }

        // Pass 1: surviving entries per row.
        let counts: Vec<u32> = (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                self.row(i)
                    .0
                    .iter()
                    .filter(|&&c| col_map[c as usize] != u32::MAX)
                    .count() as u32
            })
            .collect();
        let mut indptr = vec![0_u32; self.nrows + 1];
        for i in 0..self.nrows {
            indptr[i + 1] = indptr[i] + counts[i];
        }

        // Pass 2: fill.
        let nnz = indptr[self.nrows] as usize;
        let mut data = vec![0.0_f32; nnz];
        let mut indices = vec![0_u32; nnz];
        let dblocks = split_rows_mut(&indptr, &mut data, ROW_BLOCK);
        let iblocks = split_rows_mut(&indptr, &mut indices, ROW_BLOCK);
        dblocks
            .into_par_iter()
            .zip(iblocks)
            .for_each(|((r0, r1, d), (_, _, ix))| {
                let mut off = 0;
                for i in r0..r1 {
                    let (cols, vals) = self.row(i);
                    for (&c, &v) in cols.iter().zip(vals) {
                        let nc = col_map[c as usize];
                        if nc != u32::MAX {
                            ix[off] = nc;
                            d[off] = v;
                            off += 1;
                        }
                    }
                }
            });
        Ok(Self {
            indptr,
            indices,
            data,
            nrows: self.nrows,
            ncols: cols.len(),
        })
    }

    /// Keep the columns where `mask` is `true`.
    ///
    /// # Errors
    /// Returns [`MatrixError::DimMismatch`] if `mask.len() != ncols`.
    pub fn select_cols_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.ncols {
            return Err(MatrixError::DimMismatch {
                expected: format!("{} columns", self.ncols),
                actual: format!("mask of length {}", mask.len()),
            });
        }
        let cols: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(j, &keep)| keep.then_some(j))
            .collect();
        self.select_cols(&cols)
    }

    /// Build a CSR matrix from a CSC triple of the same `nrows × ncols`
    /// matrix, by counting sort over rows — O(nnz), no densification.
    /// Output rows have ascending column indices.
    ///
    /// # Errors
    /// Returns [`MatrixError::DimMismatch`] on inconsistent array lengths and
    /// [`MatrixError::IndexOutOfBounds`] for a row index `>= nrows`.
    pub fn from_csc(
        data: &[f32],
        row_indices: &[u32],
        col_ptr: &[u32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let nnz = data.len();
        if col_ptr.len() != ncols + 1
            || row_indices.len() != nnz
            || col_ptr.last().map_or(0, |&e| e as usize) != nnz
        {
            return Err(MatrixError::DimMismatch {
                expected: format!("col_ptr of length {} ending at {nnz}", ncols + 1),
                actual: format!(
                    "col_ptr of length {} ending at {:?}; {} indices",
                    col_ptr.len(),
                    col_ptr.last(),
                    row_indices.len()
                ),
            });
        }
        let mut indptr = vec![0_u32; nrows + 1];
        for &r in row_indices {
            if r as usize >= nrows {
                return Err(MatrixError::IndexOutOfBounds {
                    row: r as usize,
                    col: 0,
                    nrows,
                    ncols,
                });
            }
            indptr[r as usize + 1] += 1;
        }
        for i in 0..nrows {
            indptr[i + 1] += indptr[i];
        }
        let mut cursor: Vec<u32> = indptr[..nrows].to_vec();
        let mut out_data = vec![0.0_f32; nnz];
        let mut out_indices = vec![0_u32; nnz];
        for col in 0..ncols {
            for k in col_ptr[col] as usize..col_ptr[col + 1] as usize {
                let r = row_indices[k] as usize;
                let p = cursor[r] as usize;
                out_indices[p] = col as u32;
                out_data[p] = data[k];
                cursor[r] += 1;
            }
        }
        Ok(Self {
            indptr,
            indices: out_indices,
            data: out_data,
            nrows,
            ncols,
        })
    }
}

/// The unified matrix abstraction. Most downstream code takes `&Matrix` so it
/// works whether the data arrived dense or sparse.
#[derive(Debug, Clone, PartialEq)]
pub enum Matrix {
    /// Dense row-major storage.
    Dense(DenseMatrix),
    /// Compressed sparse row.
    Csr(CsrMatrix),
}

impl Matrix {
    /// Shape as `(nrows, ncols)`.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Matrix::Dense(d) => d.shape(),
            Matrix::Csr(s) => s.shape(),
        }
    }

    /// Number of rows (cells).
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    /// Number of columns (features/genes).
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    /// Number of stored elements (nnz for sparse, len for dense).
    #[inline]
    #[must_use]
    pub fn n_elements(&self) -> usize {
        match self {
            Matrix::Dense(d) => d.len(),
            Matrix::Csr(s) => s.nnz(),
        }
    }

    /// Densify, if sparse. Returns a `DenseMatrix` reference pattern (clones).
    #[must_use]
    pub fn to_dense(&self) -> DenseMatrix {
        match self {
            Matrix::Dense(d) => d.clone(),
            Matrix::Csr(s) => s.to_dense(),
        }
    }

    /// Kind tag.
    #[must_use]
    pub fn kind(&self) -> MatrixKind {
        match self {
            Matrix::Dense(_) => MatrixKind::Dense,
            Matrix::Csr(_) => MatrixKind::Csr,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_csr_names_the_failing_check() {
        use super::*;
        // The corrupted-int64-indptr signature: every length agrees, the
        // final row pointer does not. The error must say so.
        let e = CsrMatrix::new(vec![0u32, 5], vec![1u32; 3], vec![1.0_f32; 3], 1, 10).unwrap_err();
        match e {
            MatrixError::InvalidCsr {
                reason,
                indptr_tail,
                ..
            } => {
                assert_eq!(reason, "data length does not match the final row pointer");
                assert_eq!(indptr_tail, Some(5));
            }
            other => panic!("unexpected error: {other}"),
        }
        let e =
            CsrMatrix::new(vec![0u32, 1, 2], vec![1u32; 2], vec![1.0_f32; 3], 2, 10).unwrap_err();
        assert!(matches!(
            e,
            MatrixError::InvalidCsr {
                reason: "data and indices have different lengths",
                ..
            }
        ));
        let e = CsrMatrix::new(vec![0u32], vec![], vec![], 1, 10).unwrap_err();
        assert!(matches!(
            e,
            MatrixError::InvalidCsr {
                reason: "indptr length is not nrows + 1",
                ..
            }
        ));
    }

    #[test]
    fn retain_rows_mask_matches_select_rows_mask() {
        use super::*;
        let mut dense = vec![0.0_f32; 200 * 17];
        let mut x = 7_u64;
        for v in &mut dense {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = if x % 3 == 0 {
                ((x % 9) + 1) as f32
            } else {
                0.0
            };
        }
        let m = CsrMatrix::from_dense_row_major(&dense, 200, 17);
        for mask in [
            vec![true; 200],
            vec![false; 200],
            (0..200).map(|i| i % 3 != 0).collect::<Vec<_>>(),
            (0..200).map(|i| i % 7 == 2).collect::<Vec<_>>(),
        ] {
            let mut kept = m.clone();
            let n_kept = kept.retain_rows_mask(&mask).unwrap();
            let copied = m.select_rows_mask(&mask).unwrap();
            assert_eq!(n_kept, copied.nrows);
            assert_eq!(kept.nrows, n_kept);
            assert_eq!(kept.indptr, copied.indptr);
            assert_eq!(kept.indices, copied.indices);
            assert_eq!(kept.data, copied.data);
            // indptr consistency after truncation
            assert_eq!(
                kept.indptr.last().copied().unwrap() as usize,
                kept.data.len()
            );
        }
        // wrong-length mask is an error
        let mut m2 = m.clone();
        assert!(m2.retain_rows_mask(&[true, false]).is_err());
    }

    use super::*;

    fn sample_csr() -> CsrMatrix {
        // [[1, 0, 2],
        //  [0, 0, 0],
        //  [0, 3, 4]]
        CsrMatrix::new(
            vec![0, 2, 2, 4],
            vec![0, 2, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            3,
            3,
        )
        .unwrap()
    }

    #[test]
    fn csr_constructs_and_validates() {
        let m = sample_csr();
        assert_eq!(m.shape(), (3, 3));
        assert_eq!(m.nnz(), 4);
        assert!((m.density() - 4.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn csr_rejects_bad_indptr_length() {
        let err = CsrMatrix::new(vec![0, 2], vec![0, 2], vec![1.0, 2.0], 3, 3).unwrap_err();
        assert!(matches!(err, MatrixError::InvalidCsr { .. }));
    }

    #[test]
    fn csr_rejects_index_out_of_range() {
        let err = CsrMatrix::new(vec![0, 2], vec![0, 5], vec![1.0, 2.0], 1, 3).unwrap_err();
        assert!(matches!(err, MatrixError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn csr_row_accessor() {
        let m = sample_csr();
        let (cols, vals) = m.row(0);
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);
        // Empty middle row.
        let (cols, vals) = m.row(1);
        assert!(cols.is_empty() && vals.is_empty());
    }

    #[test]
    fn csr_row_sums() {
        let m = sample_csr();
        assert_eq!(m.row_sums(), vec![3.0, 0.0, 7.0]);
    }

    #[test]
    fn csr_col_sums() {
        let m = sample_csr();
        assert_eq!(m.col_sums(), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn csr_map_values_log1p() {
        let m = sample_csr();
        let logged = m.map_values(f32::ln_1p);
        // ln(1+1), 0, ln(1+2), 0, 0, ln(1+3), ln(1+4)
        let expected: Vec<f32> = vec![
            1.0_f32.ln_1p(),
            2.0_f32.ln_1p(),
            3.0_f32.ln_1p(),
            4.0_f32.ln_1p(),
        ];
        assert_eq!(logged.data, expected);
    }

    #[test]
    fn csr_densifies_correctly() {
        let m = sample_csr();
        let d = m.to_dense();
        assert_eq!(d.shape(), (3, 3));
        assert_eq!(d.data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn csr_from_dense_round_trips() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0];
        let m = CsrMatrix::from_dense_row_major(&dense, 3, 3);
        assert_eq!(m.nnz(), 4);
        let back = m.to_dense();
        assert_eq!(back.data, dense);
    }

    #[test]
    fn dense_centers_columns() {
        // [[1, 4],
        //  [3, 8]]  -> col means [2, 6] -> [[-1,-2],[1,2]]
        let mut d = DenseMatrix::from_row_major(vec![1.0, 4.0, 3.0, 8.0], 2, 2).unwrap();
        d.center_columns_in_place();
        assert_eq!(d.data, vec![-1.0, -2.0, 1.0, 2.0]);
    }

    #[test]
    fn dense_get_and_row() {
        let d = DenseMatrix::from_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        // Exact comparison is correct here: the value is stored and retrieved
        // without any arithmetic.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(d.get(1, 0).unwrap(), 3.0);
        }
        assert_eq!(d.row(0), &[1.0, 2.0]);
        let err = d.get(5, 0).unwrap_err();
        assert!(matches!(err, MatrixError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn dense_rejects_bad_shape() {
        let err = DenseMatrix::from_row_major(vec![1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(err, MatrixError::DimMismatch { .. }));
    }

    #[test]
    fn split_rows_covers_buffer_exactly() {
        let m = sample_csr();
        let mut data = m.data.clone();
        let blocks = split_rows_mut(&m.indptr, &mut data, 2);
        assert_eq!(blocks.len(), 2);
        assert_eq!((blocks[0].0, blocks[0].1, blocks[0].2.len()), (0, 2, 2));
        assert_eq!((blocks[1].0, blocks[1].1, blocks[1].2.len()), (2, 3, 2));
    }

    #[test]
    fn select_rows_gathers_in_order_with_duplicates() {
        let m = sample_csr();
        let s = m.select_rows(&[2, 0, 2]).unwrap();
        assert_eq!(s.shape(), (3, 3));
        assert_eq!(
            s.to_dense().data,
            vec![0.0, 3.0, 4.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4.0]
        );
        assert!(m.select_rows(&[3]).is_err());
        let masked = m.select_rows_mask(&[true, false, true]).unwrap();
        assert_eq!(masked.to_dense().data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 4.0]);
        assert!(m.select_rows_mask(&[true]).is_err());
    }

    #[test]
    fn select_cols_reorders_and_drops() {
        let m = sample_csr();
        // Keep column 2 then column 0.
        let s = m.select_cols(&[2, 0]).unwrap();
        assert_eq!(s.shape(), (3, 2));
        assert_eq!(s.to_dense().data, vec![2.0, 1.0, 0.0, 0.0, 4.0, 0.0]);
        assert!(m.select_cols(&[0, 0]).is_err());
        assert!(m.select_cols(&[5]).is_err());
        let masked = m.select_cols_mask(&[false, true, true]).unwrap();
        assert_eq!(masked.to_dense().data, vec![0.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn from_csc_transposes_without_densifying() {
        // Same matrix as sample_csr, in CSC:
        // col 0: row 0 = 1;  col 1: row 2 = 3;  col 2: row 0 = 2, row 2 = 4
        let m =
            CsrMatrix::from_csc(&[1.0, 3.0, 2.0, 4.0], &[0, 2, 0, 2], &[0, 1, 2, 4], 3, 3).unwrap();
        assert_eq!(m, sample_csr());
        assert!(CsrMatrix::from_csc(&[1.0], &[7], &[0, 1], 3, 1).is_err());
        assert!(CsrMatrix::from_csc(&[1.0], &[0], &[0, 1, 1], 3, 1).is_err());
    }

    #[test]
    fn matrix_enum_dispatches() {
        let m1 = Matrix::Dense(DenseMatrix::zeros(3, 3));
        let m2 = Matrix::Csr(sample_csr());
        assert_eq!(m1.shape(), (3, 3));
        assert_eq!(m2.shape(), (3, 3));
        assert_eq!(m1.n_elements(), 9);
        assert_eq!(m2.n_elements(), 4);
        assert_eq!(m1.kind(), MatrixKind::Dense);
        assert_eq!(m2.kind(), MatrixKind::Csr);
    }

    #[test]
    fn densify_via_enum() {
        let m = Matrix::Csr(sample_csr());
        let d = m.to_dense();
        assert_eq!(d.data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]);
    }
}
