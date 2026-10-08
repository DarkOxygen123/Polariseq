//! Streaming a `.h5ad` matrix as [`RowBlocks`] — the out-of-core source.
//!
//! The whole point of phase 6 is that a kernel should not care where its
//! rows come from. [`H5adStream`] makes an on-disk `.h5ad` look like any
//! other block source, so `qc_metrics`, `gene_mean_var`, `gram_matrix`,
//! `spmm`/`spmm_t` — and therefore PCA and HVG — run on a file that never
//! fits in RAM, using the same code (and producing bit-identical results)
//! as the in-memory path.
//!
//! ## Why reads are serialised
//!
//! The vendored HDF5 is **not thread-safe** (handbook chapter 00, pitfall
//! 13). Blocks are therefore read one at a time on the calling thread; the
//! *work* is what runs in parallel. Each pass reads a batch of up to
//! [`PREFETCH`] blocks sequentially, then hands the batch to rayon and
//! calls `f` on each block concurrently. Peak memory is bounded by
//! `PREFETCH × rows_per_block` rows of nonzeros, not by the file.
//!
//! That ordering is also why this is fast enough to be worth doing: the
//! read is one sequential scan of the compressed chunks and the decode
//! overlaps the next read only across batches, but the alternative — a
//! parallel read — is a crash, not a speedup.

use polariseq_core::rowblocks::RowBlocks;
use polariseq_core::rsvd::CsrRef;
use polariseq_core::CsrMatrix;
use rayon::prelude::*;

use crate::lazy::{read_rows_any, scan_any, DatasetMeta};
use crate::Result;

/// Blocks read per batch before the batch is processed in parallel, when the
/// caller does not say otherwise.
const DEFAULT_PREFETCH: usize = 4;

/// Rows per block when the caller does not say otherwise.
const DEFAULT_MAX_BLOCK_ROWS: usize = 32_768;

/// A `.h5ad` (or 10x `.h5` / `.mtx`) file presented as row blocks.
///
/// Construction only scans metadata — no matrix data is read until a
/// kernel asks for blocks, and nothing is retained between passes.
pub struct H5adStream {
    path: String,
    meta: DatasetMeta,
    max_block_rows: usize,
    prefetch: usize,
}

/// How much of a stream may be in flight at once.
///
/// The cap lives on the **source**, not on the kernels: a kernel asks for
/// whatever block size suits its arithmetic, and only the source knows that
/// its blocks are decoded copies read off a disk rather than slices of a
/// resident matrix. So the source clamps.
///
/// This is not a micro-optimisation. At ~1 100 nonzeros per cell a
/// 32 768-row block is ~292 MB, and four in flight is 1.17 GB of pure
/// working space — more than some machines have free. `ps.plan()` solves
/// for the rows that fit the budget and passes the answer here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamLimits {
    /// Upper bound on rows per block, whatever the kernel requests.
    pub max_block_rows: usize,
    /// Blocks read (and therefore held) per batch.
    pub prefetch: usize,
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_block_rows: DEFAULT_MAX_BLOCK_ROWS,
            prefetch: DEFAULT_PREFETCH,
        }
    }
}

impl H5adStream {
    /// Scan `path` and prepare it as a block source.
    ///
    /// # Errors
    /// Returns [`crate::IoError`] if the file cannot be scanned.
    pub fn open(path: &str) -> Result<Self> {
        let meta = scan_any(path)?;
        let limits = StreamLimits::default();
        Ok(Self {
            path: path.to_string(),
            meta,
            max_block_rows: limits.max_block_rows,
            prefetch: limits.prefetch,
        })
    }

    /// Bound how much of this stream may be resident at once.
    #[must_use]
    pub fn with_limits(mut self, limits: StreamLimits) -> Self {
        self.max_block_rows = limits.max_block_rows.max(1);
        self.prefetch = limits.prefetch.max(1);
        self
    }

    /// The limits in force.
    #[must_use]
    pub fn limits(&self) -> StreamLimits {
        StreamLimits {
            max_block_rows: self.max_block_rows,
            prefetch: self.prefetch,
        }
    }

    /// The scanned metadata (shape, nnz, names, estimated sizes).
    #[must_use]
    pub fn meta(&self) -> &DatasetMeta {
        &self.meta
    }

    /// Path this stream reads from.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Read one row range. Errors are surfaced by the caller's error slot
    /// rather than by the `RowBlocks` contract, which cannot fail.
    fn read_block(&self, r0: usize, r1: usize) -> Result<CsrMatrix> {
        read_rows_any(&self.path, r0, r1)
    }
}

impl RowBlocks for H5adStream {
    fn n_rows(&self) -> usize {
        self.meta.n_cells
    }

    fn n_cols(&self) -> usize {
        self.meta.n_genes
    }

    /// Reads sequentially in batches (HDF5 is not thread-safe), then runs
    /// `f` over each batch in parallel. A read failure mid-pass stops the
    /// iteration; the block is simply not visited, and
    /// [`H5adStream::run`] is the checked entry point that reports it.
    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        let _ = self.run(rows_per_block, f);
    }
}

impl H5adStream {
    /// [`RowBlocks::for_each_block`] with the read errors kept.
    ///
    /// # Errors
    /// Returns the first read failure; blocks before it have been visited.
    pub fn run(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) -> Result<()> {
        let n = self.n_rows();
        // The kernel asks; the source caps. See `StreamLimits`.
        let rpb = rows_per_block.clamp(1, self.max_block_rows.max(1));
        let n_blocks = n.div_ceil(rpb);
        let mut b0 = 0;
        while b0 < n_blocks {
            let b1 = (b0 + self.prefetch).min(n_blocks);
            // Sequential reads: one HDF5 caller at a time.
            let mut batch = Vec::with_capacity(b1 - b0);
            for b in b0..b1 {
                let (r0, r1) = (b * rpb, ((b + 1) * rpb).min(n));
                batch.push((r0, self.read_block(r0, r1)?));
            }
            // Parallel compute over the batch just read.
            batch.par_iter().for_each(|(r0, m)| {
                f(
                    *r0,
                    CsrRef {
                        data: &m.data,
                        indices: &m.indices,
                        indptr: &m.indptr,
                        n_rows: m.nrows,
                        n_cols: m.ncols,
                    },
                );
            });
            b0 = b1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polariseq_core::preprocess::{gene_mean_var_blocks, qc_metrics_blocks};
    use polariseq_core::rsvd::{column_means_blocks, gram_matrix_blocks};
    use std::sync::Mutex;

    fn fixture() -> String {
        format!(
            "{}/../../tests/data/sparse_100x200.h5ad",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// `(row offset, indptr, indices, data)` captured from one block.
    type CapturedBlock = (usize, Vec<u32>, Vec<u32>, Vec<f32>);

    fn in_ram() -> CsrMatrix {
        let f = crate::AnnDataFile::open(fixture()).unwrap();
        match f.read_x().unwrap() {
            polariseq_core::Matrix::Csr(c) => c,
            polariseq_core::Matrix::Dense(_) => unreachable!("fixture is sparse"),
        }
    }

    #[test]
    fn shape_matches_the_file() {
        let s = H5adStream::open(&fixture()).unwrap();
        assert_eq!((s.n_rows(), s.n_cols()), (100, 200));
    }

    /// Blocks must tile the matrix exactly, for block sizes that divide the
    /// row count and sizes that do not. Reassembling every block in row
    /// order must reproduce the in-RAM matrix entry for entry.
    #[test]
    fn blocks_reassemble_the_whole_matrix() {
        let stream = H5adStream::open(&fixture()).unwrap();
        let ram = in_ram();
        for rpb in [1, 7, 32, 99, 100, 4096] {
            let seen: Mutex<Vec<CapturedBlock>> = Mutex::new(Vec::new());
            stream
                .run(rpb, &|off, blk| {
                    seen.lock().unwrap().push((
                        off,
                        blk.indptr.to_vec(),
                        blk.indices.to_vec(),
                        blk.data.to_vec(),
                    ));
                })
                .unwrap();
            let mut blocks = seen.into_inner().unwrap();
            blocks.sort_unstable_by_key(|b| b.0);
            assert_eq!(
                blocks.len(),
                100_usize.div_ceil(rpb.clamp(1, 100)),
                "block count at rpb={rpb}"
            );

            let mut row = 0usize;
            for (off, indptr, indices, data) in blocks {
                assert_eq!(off, row, "blocks out of order at rpb={rpb}");
                for i in 0..indptr.len() - 1 {
                    let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
                    let (want_c, want_v) = ram.row(row);
                    assert_eq!(&indices[s..e], want_c, "row {row} indices at rpb={rpb}");
                    assert_eq!(&data[s..e], want_v, "row {row} values at rpb={rpb}");
                    row += 1;
                }
            }
            assert_eq!(row, 100, "rows covered at rpb={rpb}");
        }
    }

    /// The point of the abstraction: identical kernel output from a streamed
    /// file and the same matrix in RAM — not close, identical.
    #[test]
    fn kernels_match_the_in_ram_matrix() {
        let stream = H5adStream::open(&fixture()).unwrap();
        let ram = in_ram();
        let mito: Vec<bool> = (0..200).map(|i| i % 17 == 0).collect();

        let a = qc_metrics_blocks(&stream, Some(&mito));
        let b = qc_metrics_blocks(&ram, Some(&mito));
        assert_eq!(a, b, "qc_metrics");

        for expm1 in [false, true] {
            assert_eq!(
                gene_mean_var_blocks(&stream, expm1),
                gene_mean_var_blocks(&ram, expm1),
                "gene_mean_var(expm1={expm1})"
            );
        }
        assert_eq!(
            column_means_blocks(&stream),
            column_means_blocks(&ram),
            "column_means"
        );
        assert_eq!(
            gram_matrix_blocks(&stream),
            gram_matrix_blocks(&ram),
            "gram_matrix"
        );
    }

    #[test]
    fn reports_a_bad_path() {
        assert!(H5adStream::open("/nonexistent/never.h5ad").is_err());
    }
}
