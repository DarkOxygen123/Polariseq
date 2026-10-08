#![allow(
    unsafe_code,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown
)]
//! Disk-backed sparse matrix (DBCSR) for out-of-core processing.
//!
//! When a dataset is too large for RAM, Polariseq extracts the CSR triple
//! (data, indices, indptr) to flat binary files on SSD, then memory-maps them.
//! The OS page cache transparently manages which pages stay in RAM — hot
//! pages (recently accessed rows) are cached; cold pages are evicted to SSD.
//!
//! This lets us process matrices larger than RAM with near-RAM speed for
//! sequential access patterns (which is what our streaming algorithms use).
//!
//! ## Resource limits
//!
//! The user sets `max_ram_gb` and `max_ssd_gb`. The system checks:
//! - If the dataset fits in `max_ram_gb` → load into RAM (fast path)
//! - If it fits in `max_ssd_gb` → extract to SSD + mmap (streaming path)
//! - If it exceeds both → error
//!
//! ## Layout
//!
//! A DBCSR is stored as three files:
//! - `{prefix}.data`    — `nnz × f32` (4 bytes each)
//! - `{prefix}.indices` — `nnz × u32` (4 bytes each)
//! - `{prefix}.indptr`  — `(nrows+1) × u32` (4 bytes each)

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;

/// A CSR matrix backed by memory-mapped files on SSD.
///
/// The data stays on disk; only accessed pages are in RAM (managed by the OS
/// page cache). This enables processing matrices larger than available RAM.
pub struct DiskBackedCSR {
    /// Memory-mapped data array (read-write for in-place transforms).
    data_mmap: MmapMut,
    /// Memory-mapped column indices.
    indices_mmap: Mmap,
    /// Memory-mapped row pointers.
    indptr_mmap: Mmap,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Number of stored nonzeros.
    pub nnz: usize,
    /// The file prefix (for cleanup).
    prefix: PathBuf,
    /// Whether to delete files on drop.
    delete_on_drop: bool,
}

impl DiskBackedCSR {
    /// Create a DBCSR from an in-memory CSR triple, writing to SSD.
    ///
    /// Writes three flat binary files: `{prefix}.data`, `{prefix}.indices`,
    /// `{prefix}.indptr`. Then memory-maps them for read-write access.
    pub fn from_csr(
        data: &[f32],
        indices: &[u32],
        indptr: &[u32],
        nrows: usize,
        ncols: usize,
        prefix: &Path,
    ) -> std::io::Result<Self> {
        let nnz = data.len();
        std::fs::create_dir_all(prefix.parent().unwrap_or(Path::new(".")))?;

        let data_path = prefix.with_extension("data");
        let indices_path = prefix.with_extension("indices");
        let indptr_path = prefix.with_extension("indptr");

        // Write data (read-write for in-place transforms like normalize/log1p).
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&data_path)?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&bytes)?;
            f.flush()?;
        }
        // Write indices (read-only).
        {
            let mut f = File::create(&indices_path)?;
            let bytes: Vec<u8> = indices.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&bytes)?;
        }
        // Write indptr (read-only).
        {
            let mut f = File::create(&indptr_path)?;
            let bytes: Vec<u8> = indptr.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&bytes)?;
        }

        Self::open(
            &data_path,
            &indices_path,
            &indptr_path,
            nrows,
            ncols,
            nnz,
            prefix.to_path_buf(),
            true,
        )
    }

    /// Open existing DBCSR files and memory-map them.
    #[allow(clippy::too_many_arguments)]
    fn open(
        data_path: &Path,
        indices_path: &Path,
        indptr_path: &Path,
        nrows: usize,
        ncols: usize,
        nnz: usize,
        prefix: PathBuf,
        delete_on_drop: bool,
    ) -> std::io::Result<Self> {
        let data_file = OpenOptions::new().read(true).write(true).open(data_path)?;
        let indices_file = File::open(indices_path)?;
        let indptr_file = File::open(indptr_path)?;

        let data_mmap = unsafe { MmapMut::map_mut(&data_file)? };
        let indices_mmap = unsafe { Mmap::map(&indices_file)? };
        let indptr_mmap = unsafe { Mmap::map(&indptr_file)? };

        Ok(Self {
            data_mmap,
            indices_mmap,
            indptr_mmap,
            nrows,
            ncols,
            nnz,
            prefix,
            delete_on_drop,
        })
    }

    /// View the data array as `&mut [f32]` (for in-place transforms).
    pub fn data_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data_mmap.as_mut_ptr() as *mut f32, self.nnz) }
    }

    /// View the data array as `&[f32]` (read-only).
    pub fn data(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data_mmap.as_ptr() as *const f32, self.nnz) }
    }

    /// View the indices array as `&[u32]`.
    pub fn indices(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.indices_mmap.as_ptr() as *const u32, self.nnz) }
    }

    /// View the indptr array as `&[u32]`.
    pub fn indptr(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(self.indptr_mmap.as_ptr() as *const u32, self.nrows + 1)
        }
    }

    /// Get the row range `[start, end)` in the data/indices arrays for row `i`.
    pub fn row_range(&self, i: usize) -> (usize, usize) {
        let indptr = self.indptr();
        (indptr[i] as usize, indptr[i + 1] as usize)
    }

    /// Apply a function to every nonzero value in place (streaming through SSD).
    /// This is the foundation for normalize_total, log1p, scale.
    pub fn map_values_in_place<F>(&mut self, f: F)
    where
        F: Fn(f32) -> f32 + Sync,
    {
        let data = self.data_mut();
        data.par_iter_mut().for_each(|x| *x = f(*x));
        // Flush dirty pages to disk so subsequent reads see the changes.
        let _ = self.data_mmap.flush();
    }

    /// Compute per-row sums (for normalize_total). Streams through SSD.
    pub fn row_sums(&self) -> Vec<f32> {
        let data = self.data();
        let indptr = self.indptr();
        (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
                data[s..e].iter().sum()
            })
            .collect()
    }

    /// Compute per-column sums. Two-pass: first compute, then reduce.
    pub fn col_sums(&self) -> Vec<f32> {
        let data = self.data();
        let indices = self.indices();
        let mut sums = vec![0.0_f32; self.ncols];
        for k in 0..self.nnz {
            sums[indices[k] as usize] += data[k];
        }
        sums
    }

    /// Compute per-column sum of squares (for HVG variance).
    pub fn col_sq_sums(&self) -> Vec<f64> {
        let data = self.data();
        let indices = self.indices();
        let mut sums = vec![0.0_f64; self.ncols];
        for k in 0..self.nnz {
            let v = f64::from(data[k]);
            sums[indices[k] as usize] += v * v;
        }
        sums
    }

    /// Subset to selected columns (for HVG selection). Creates a NEW DBCSR.
    ///
    /// # Panics
    /// Panics if the mask length does not match `ncols`.
    pub fn select_columns(&self, mask: &[bool], new_prefix: &Path) -> std::io::Result<Self> {
        assert_eq!(mask.len(), self.ncols);
        let col_map: Vec<Option<u32>> = mask
            .iter()
            .scan(0u32, |next, &keep| {
                if keep {
                    let v = Some(*next);
                    *next += 1;
                    Some(v)
                } else {
                    Some(None)
                }
            })
            .collect();
        let new_ncols = col_map.iter().filter(|c| c.is_some()).count();

        let data = self.data();
        let indices = self.indices();
        let indptr = self.indptr();

        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0_u32; self.nrows + 1];

        for i in 0..self.nrows {
            let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
            for k in s..e {
                if let Some(new_col) = col_map[indices[k] as usize] {
                    new_data.push(data[k]);
                    new_indices.push(new_col);
                }
            }
            new_indptr[i + 1] = new_data.len() as u32;
        }

        Self::from_csr(
            &new_data,
            &new_indices,
            &new_indptr,
            self.nrows,
            new_ncols,
            new_prefix,
        )
    }

    /// Scale each row's values by a per-row factor (for normalize_total).
    pub fn scale_rows(&mut self, scales: &[f32]) {
        // Use raw pointers to avoid the borrow checker conflict between
        // data_mut (mutable) and indptr (immutable) — both borrow self.
        let data_ptr = self.data_mmap.as_mut_ptr() as *mut f32;
        let indptr_ptr = self.indptr_mmap.as_ptr() as *const u32;
        let nrows = self.nrows;
        for (i, &scale) in scales.iter().take(nrows).enumerate() {
            let s = unsafe { *indptr_ptr.add(i) } as usize;
            let e = unsafe { *indptr_ptr.add(i + 1) } as usize;

            for k in s..e {
                unsafe {
                    *data_ptr.add(k) *= scale;
                }
            }
        }
        let _ = self.data_mmap.flush();
    }

    /// Sparse-dense matrix product: `self × B` where B is `(ncols × l)`.
    /// Returns `(nrows × l)` dense result. This is the core of randomized SVD
    /// on disk-backed data — streams through SSD row-by-row.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dense(&self, b: &[f32], l: usize) -> Vec<f32> {
        let data = self.data();
        let indices = self.indices();
        let indptr = self.indptr();

        let mut result = vec![0.0_f32; self.nrows * l];
        result
            .par_chunks_mut(l)
            .enumerate()
            .for_each(|(i, row_out)| {
                let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
                for k in s..e {
                    let col = indices[k] as usize;
                    let val = data[k];
                    let b_row = &b[col * l..(col + 1) * l];
                    for (out, &bv) in row_out.iter_mut().zip(b_row) {
                        *out += val * bv;
                    }
                }
            });
        result
    }

    /// Disk space used by this DBCSR (in bytes).
    pub fn disk_size_bytes(&self) -> usize {
        self.nnz * 8 + (self.nrows + 1) * 4 // data(4) + indices(4) + indptr(4)
    }

    /// Open an existing DBCSR triple written by [`Self::from_csr`], without
    /// deleting the files on drop. `nnz` is taken from the data file's size.
    ///
    /// This is the read side of spill-to-SSD: a matrix written once (while
    /// loading) and then streamed through the kernels as a [`RowBlocks`]
    /// source.
    pub fn open_at(prefix: &Path, nrows: usize, ncols: usize) -> std::io::Result<Self> {
        let data_path = prefix.with_extension("data");
        let nnz = std::fs::metadata(&data_path)?.len() as usize / 4;
        Self::open(
            &data_path,
            &prefix.with_extension("indices"),
            &prefix.with_extension("indptr"),
            nrows,
            ncols,
            nnz,
            prefix.to_path_buf(),
            false,
        )
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> std::io::Result<()> {
        self.data_mmap.flush()?;
        Ok(())
    }
}

impl Drop for DiskBackedCSR {
    fn drop(&mut self) {
        let _ = self.data_mmap.flush();
        if self.delete_on_drop {
            let _ = std::fs::remove_file(self.prefix.with_extension("data"));
            let _ = std::fs::remove_file(self.prefix.with_extension("indices"));
            let _ = std::fs::remove_file(self.prefix.with_extension("indptr"));
        }
    }
}

/// Resource limits for out-of-core processing.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum RAM to use (in bytes) for in-memory operations.
    pub max_ram_bytes: usize,
    /// Maximum SSD space (in bytes) for spill files.
    pub max_ssd_bytes: usize,
    /// Directory for spill files.
    pub spill_dir: PathBuf,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_ram_bytes: 8 * 1_073_741_824,   // 8 GB
            max_ssd_bytes: 100 * 1_073_741_824, // 100 GB
            spill_dir: std::env::temp_dir().join("polariseq"),
        }
    }
}

impl ResourceLimits {
    /// Create from GB values (the user-facing API).
    #[must_use]
    pub fn from_gb(max_ram_gb: f64, max_ssd_gb: f64, spill_dir: &str) -> Self {
        Self {
            max_ram_bytes: (max_ram_gb * 1_073_741_824.0) as usize,
            max_ssd_bytes: (max_ssd_gb * 1_073_741_824.0) as usize,
            spill_dir: PathBuf::from(spill_dir),
        }
    }

    /// Check if a CSR matrix of `nnz` nonzeros fits in RAM.
    #[must_use]
    pub fn fits_in_ram(&self, nnz: usize, nrows: usize) -> bool {
        let bytes = nnz * 8 + (nrows + 1) * 4;
        bytes <= self.max_ram_bytes
    }

    /// Check if it fits on SSD.
    #[must_use]
    pub fn fits_on_ssd(&self, nnz: usize, nrows: usize) -> bool {
        let bytes = nnz * 8 + (nrows + 1) * 4;
        bytes <= self.max_ssd_bytes
    }
}

/// The mmap'd matrix as a block source: blocks are slices of the mapped
/// arrays, so nothing is copied and the OS page cache decides what stays
/// resident. Sequential block order gives the kernel a sequential read
/// pattern, which is what makes an SSD-backed pass fast.
impl crate::rowblocks::RowBlocks for DiskBackedCSR {
    fn n_rows(&self) -> usize {
        self.nrows
    }

    fn n_cols(&self) -> usize {
        self.ncols
    }

    fn for_each_block(
        &self,
        rows_per_block: usize,
        f: &(dyn Fn(usize, crate::rsvd::CsrRef<'_>) + Sync),
    ) {
        let (data, indices, indptr) = (self.data(), self.indices(), self.indptr());
        let rpb = rows_per_block.max(1);
        let n_blocks = self.nrows.div_ceil(rpb);
        (0..n_blocks).into_par_iter().for_each(|b| {
            let (r0, r1) = (b * rpb, ((b + 1) * rpb).min(self.nrows));
            let (s, e) = (indptr[r0] as usize, indptr[r1] as usize);
            let local: Vec<u32> = indptr[r0..=r1].iter().map(|&v| v - indptr[r0]).collect();
            f(
                r0,
                crate::rsvd::CsrRef {
                    data: &data[s..e],
                    indices: &indices[s..e],
                    indptr: &local,
                    n_rows: r1 - r0,
                    n_cols: self.ncols,
                },
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbcsr_round_trip() {
        let tmp = std::env::temp_dir().join("polariseq_test_dbcsr");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let indices = vec![0_u32, 2, 1, 2];
        let indptr = vec![0_u32, 2, 4];
        let prefix = tmp.join("test");

        let mut db = DiskBackedCSR::from_csr(&data, &indices, &indptr, 2, 3, &prefix).unwrap();
        assert_eq!(db.nrows, 2);
        assert_eq!(db.ncols, 3);
        assert_eq!(db.nnz, 4);

        // Data is accessible.
        assert_eq!(db.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(db.indices(), &[0, 2, 1, 2]);

        // Row sums.
        let sums = db.row_sums();
        assert_eq!(sums, vec![3.0, 7.0]);

        // In-place log1p.
        db.map_values_in_place(f32::ln_1p);
        assert!((db.data()[0] - 1.0_f32.ln_1p()).abs() < 1e-6);
    }

    #[test]
    fn dbcsr_col_sums() {
        let tmp = std::env::temp_dir().join("polariseq_test_dbcsr2");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0_u32, 2, 1, 0, 2];
        let indptr = vec![0_u32, 2, 3, 5];
        let prefix = tmp.join("test2");

        let db = DiskBackedCSR::from_csr(&data, &indices, &indptr, 3, 3, &prefix).unwrap();
        assert_eq!(db.col_sums(), vec![5.0, 3.0, 7.0]); // col 0: 1+4=5, col 1: 3, col 2: 2+5=7
    }

    #[test]
    fn dbcsr_matmul() {
        let tmp = std::env::temp_dir().join("polariseq_test_dbcsr3");
        // [[1, 0, 2], [0, 3, 0]] × [[1, 0], [0, 1], [1, 1]] = [[3, 2], [0, 3]]
        let data = vec![1.0_f32, 2.0, 3.0];
        let indices = vec![0_u32, 2, 1];
        let indptr = vec![0_u32, 2, 3];
        let prefix = tmp.join("test3");
        let db = DiskBackedCSR::from_csr(&data, &indices, &indptr, 2, 3, &prefix).unwrap();
        let b = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2
        let result = db.matmul_dense(&b, 2);
        // Row 0: [1*1+2*1, 1*0+2*1] = [3, 2]
        // Row 1: [3*0, 3*1] = [0, 3]
        assert_eq!(&result, &[3.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn dbcsr_select_columns() {
        let tmp = std::env::temp_dir().join("polariseq_test_dbcsr4");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let indices = vec![0_u32, 2, 1, 2];
        let indptr = vec![0_u32, 2, 4];
        let prefix = tmp.join("test4");
        let db = DiskBackedCSR::from_csr(&data, &indices, &indptr, 2, 3, &prefix).unwrap();
        // Keep cols 0 and 2, drop col 1.
        let mask = vec![true, false, true];
        let sub = db.select_columns(&mask, &tmp.join("test4_sub")).unwrap();
        assert_eq!(sub.ncols, 2);
        assert_eq!(sub.nnz, 3); // col 1 (value 3.0) dropped
        assert_eq!(sub.data(), &[1.0, 2.0, 4.0]);
        assert_eq!(sub.indices(), &[0, 1, 1]); // remapped: 0→0, 2→1
    }

    /// The mmap'd matrix must behave as a block source exactly like the
    /// in-RAM matrix it was written from — same blocks, same kernel output.
    #[test]
    fn dbcsr_row_blocks_match_in_ram() {
        use crate::preprocess::{gene_mean_var_blocks, qc_metrics_blocks};
        use crate::rowblocks::RowBlocks;

        let tmp = std::env::temp_dir().join("polariseq_test_dbcsr_blocks");
        // 40 rows x 6 cols with empty rows and an explicit zero.
        let mut indptr = vec![0_u32];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for i in 0..40_u32 {
            if i % 5 != 0 {
                indices.extend([i % 6, (i + 2) % 6]);
                data.extend([(i % 7) as f32, if i % 11 == 0 { 0.0 } else { 1.5 }]);
            }
            indptr.push(data.len() as u32);
        }
        let ram = crate::matrix::CsrMatrix::new(indptr, indices, data, 40, 6).unwrap();
        let db = DiskBackedCSR::from_csr(
            &ram.data,
            &ram.indices,
            &ram.indptr,
            ram.nrows,
            ram.ncols,
            &tmp.join("blocks"),
        )
        .unwrap();

        assert_eq!((db.n_rows(), db.n_cols()), (40, 6));
        assert_eq!(qc_metrics_blocks(&db, None), qc_metrics_blocks(&ram, None));
        assert_eq!(
            gene_mean_var_blocks(&db, false),
            gene_mean_var_blocks(&ram, false)
        );

        // Blocks tile the matrix in row order for sizes that do and do not
        // divide the row count.
        for rpb in [1, 7, 40, 256] {
            let seen = std::sync::Mutex::new(Vec::new());
            db.for_each_block(rpb, &|off, blk| {
                seen.lock().unwrap().push((off, blk.n_rows));
            });
            let mut got = seen.into_inner().unwrap();
            got.sort_unstable();
            assert_eq!(got.iter().map(|b| b.1).sum::<usize>(), 40, "rpb={rpb}");
            assert_eq!(got[0].0, 0, "rpb={rpb}");
        }
    }

    #[test]
    fn resource_limits_fit_check() {
        let rl = ResourceLimits::from_gb(4.0, 100.0, "/tmp/test");
        // 1B nnz → 8 GB data + 4 GB indices = ~12 GB. Doesn't fit in 4 GB RAM.
        assert!(!rl.fits_in_ram(1_000_000_000, 1_000_000));
        // But fits on 100 GB SSD.
        assert!(rl.fits_on_ssd(1_000_000_000, 1_000_000));
        // 100M nnz → ~800 MB + 400 MB = ~1.2 GB. Fits in 4 GB RAM.
        assert!(rl.fits_in_ram(100_000_000, 1_000_000));
    }
}
