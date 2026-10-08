//! One abstraction for every matrix source: blocks of CSR rows.
//!
//! Every kernel that loops over rows (QC, gene stats, sparse products,
//! Gram matrix) runs on a [`RowBlocks`] source instead of a concrete
//! matrix, so in-RAM, memory-mapped and (later) streamed-from-HDF5 inputs
//! share one code path. The handbook contract:
//!
//! - Blocks cover `rows_per_block` rows in row order; `f` receives the
//!   global row offset and the block as a [`CsrRef`] with `indptr` rebased
//!   to 0 (the rebase buffer is one small `Vec` per block — blocks are few
//!   and large).
//! - Sources may invoke `f` from any thread and in any completion order;
//!   blocks are disjoint, so kernels fold partials through
//!   [`BlockFold`], which merges them **in block order** — results are
//!   bit-identical for any thread count or scheduling.

#![allow(clippy::module_name_repetitions)]

use crate::matrix::CsrMatrix;
use crate::rsvd::CsrRef;
use rayon::prelude::*;
use std::sync::Mutex;

/// A source of row blocks of a cells × genes matrix, in row order.
///
/// `Sync` because every implementation already hands `&self` to rayon inside
/// `for_each_block`, and because adapters (see
/// [`NormalizedView`](crate::preprocess::NormalizedView)) wrap a source in a
/// closure that runs on the worker threads.
pub trait RowBlocks: Sync {
    /// Total rows of the matrix.
    fn n_rows(&self) -> usize;
    /// Total columns of the matrix.
    fn n_cols(&self) -> usize;
    /// Visit blocks of `rows_per_block` rows (the last may be shorter).
    /// `f` receives the global row offset of the block start and the block
    /// itself (`indptr` rebased to 0, `n_rows` = block length).
    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync));
}

/// Row pointers for rows `r0..=r1`, rebased so the block starts at 0.
/// Returned separately from the `CsrRef` that borrows it — the caller keeps
/// both alive for the duration of the callback.
fn rebase_indptr(indptr: &[u32], r0: usize, r1: usize) -> Vec<u32> {
    let mut local = Vec::with_capacity(r1 - r0 + 1);
    for k in r0..=r1 {
        local.push(indptr[k] - indptr[r0]);
    }
    local
}

impl RowBlocks for CsrMatrix {
    fn n_rows(&self) -> usize {
        self.nrows
    }

    fn n_cols(&self) -> usize {
        self.ncols
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        let rpb = rows_per_block.max(1);
        let n_blocks = self.nrows.div_ceil(rpb);
        (0..n_blocks).into_par_iter().for_each(|b| {
            let (r0, r1) = (b * rpb, ((b + 1) * rpb).min(self.nrows));
            let (s, e) = (self.indptr[r0] as usize, self.indptr[r1] as usize);
            let local = rebase_indptr(&self.indptr, r0, r1);
            let block = CsrRef {
                data: &self.data[s..e],
                indices: &self.indices[s..e],
                indptr: &local,
                n_rows: r1 - r0,
                n_cols: self.ncols,
            };
            f(r0, block);
        });
    }
}

impl RowBlocks for CsrRef<'_> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        let rpb = rows_per_block.max(1);
        let n_blocks = self.n_rows.div_ceil(rpb);
        (0..n_blocks).into_par_iter().for_each(|b| {
            let (r0, r1) = (b * rpb, ((b + 1) * rpb).min(self.n_rows));
            let (s, e) = (self.indptr[r0] as usize, self.indptr[r1] as usize);
            let local = rebase_indptr(self.indptr, r0, r1);
            let block = CsrRef {
                data: &self.data[s..e],
                indices: &self.indices[s..e],
                indptr: &local,
                n_rows: r1 - r0,
                n_cols: self.n_cols,
            };
            f(r0, block);
        });
    }
}

/// Folds block partials into an accumulator **in block order**, whatever
/// order the blocks complete in. Partials are keyed by the row offset their
/// block started at — unique per block — and merged in ascending offset
/// order at [`BlockFold::finish`], so the result is deterministic for any
/// thread schedule.
///
/// Keying by offset (not by `offset / requested_chunk`, an earlier design)
/// matters because sources may legitimately emit **smaller** blocks than the
/// consumer asked for: `FileCsr` clamps `rows_per_block` to its own
/// budget-derived cap, which is the memory limit doing its job. Under the
/// slot design those smaller blocks collided in one slot and each `store`
/// overwrote the last — on the 10k fixture with 4 096-row blocks, only the
/// final 1 807 rows survived into the gene statistics and HVG selection
/// kept 54 of 200 genes (found by the telemetry harness; chapter 09).
///
/// Shared behind a `Mutex` because [`RowBlocks`] sources may call the
/// block callback from several threads.
pub(crate) struct BlockFold<T> {
    partials: Mutex<std::collections::BTreeMap<usize, T>>,
    merge: fn(&mut T, T),
}

impl<T: Send> BlockFold<T> {
    /// `merge` folds a later partial into the accumulator
    /// (`merge(&mut acc, partial)`); partials merge in ascending offset.
    pub(crate) fn new(merge: fn(&mut T, T)) -> Self {
        Self {
            partials: Mutex::new(std::collections::BTreeMap::new()),
            merge,
        }
    }

    /// Store the partial for the block that started at row `offset`.
    ///
    /// # Panics
    /// Panics if two blocks claim the same offset — that is a source bug,
    /// not a scheduling artifact, and must not be survived silently.
    pub(crate) fn store(&self, offset: usize, partial: T) {
        let mut map = self.partials.lock().expect("block fold poisoned");
        assert!(
            map.insert(offset, partial).is_none(),
            "BlockFold: duplicate block offset {offset}"
        );
    }

    /// Merge every stored partial into `zero`, in ascending offset order.
    /// With no partials (no block ever fired) the result is `zero`.
    pub(crate) fn finish(self, zero: T) -> T {
        let map = self.partials.into_inner().expect("block fold poisoned");
        let mut acc = zero;
        for (_, p) in map {
            (self.merge)(&mut acc, p);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_matrix_blocks_cover_all_rows_exactly_once() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let m = CsrMatrix::from_dense_row_major(
            &(0..60_000)
                .map(|i| (i % 7) as f32 * 0.5)
                .collect::<Vec<_>>(),
            3000,
            20,
        );
        let visited: Vec<AtomicU32> = (0..3000).map(|_| AtomicU32::new(0)).collect();
        m.for_each_block(1024, &|off, b| {
            assert_eq!(b.indptr.len(), b.n_rows + 1);
            assert_eq!(b.indptr[0], 0);
            let nnz: u32 = b.indptr[b.n_rows];
            assert_eq!(nnz as usize, b.data.len());
            assert_eq!(nnz as usize, b.indices.len());
            for r in 0..b.n_rows {
                visited[off + r].fetch_add(1, Ordering::Relaxed);
            }
        });
        assert!(visited.iter().all(|v| v.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn block_fold_merges_in_block_order_regardless_of_completion() {
        // Concatenating in completion order 2,0,1 must still yield "0123".
        let fold: BlockFold<String> = BlockFold::new(|acc, p| acc.push_str(&p));
        fold.store(2, "2".to_string());
        fold.store(0, "0".to_string());
        fold.store(1, "1".to_string());
        fold.store(3, "3".to_string());
        assert_eq!(fold.finish(String::new()), "0123");
    }

    /// A source that emits SMALLER blocks than the consumer asked for is a
    /// source doing its job (`FileCsr` clamps to its budget-derived cap).
    /// Under the old slot-keyed fold, every small block sharing a slot
    /// overwrote the last — silently dropping rows from reductions. Keyed by
    /// offset, arbitrary splits fold correctly and in order.
    #[test]
    fn block_fold_survives_a_source_that_splits_requested_blocks() {
        use std::sync::Mutex;

        use super::super::preprocess::gene_mean_var_blocks;

        struct Clamped {
            inner: crate::CsrMatrix,
            cap: usize,
            seen: Mutex<Vec<usize>>,
        }
        impl RowBlocks for Clamped {
            fn n_rows(&self) -> usize {
                self.inner.nrows
            }
            fn n_cols(&self) -> usize {
                self.inner.ncols
            }
            fn for_each_block(
                &self,
                rows_per_block: usize,
                f: &(dyn Fn(usize, CsrRef<'_>) + Sync),
            ) {
                let rpb = rows_per_block.min(self.cap);
                let mut r0 = 0;
                while r0 < self.inner.nrows {
                    let r1 = (r0 + rpb).min(self.inner.nrows);
                    self.seen.lock().unwrap().push(r0);
                    let (a, b) = (
                        self.inner.indptr[r0] as usize,
                        self.inner.indptr[r1] as usize,
                    );
                    let local: Vec<u32> = self.inner.indptr[r0..=r1]
                        .iter()
                        .map(|&v| v - self.inner.indptr[r0])
                        .collect();
                    f(
                        r0,
                        CsrRef {
                            data: &self.inner.data[a..b],
                            indices: &self.inner.indices[a..b],
                            indptr: &local,
                            n_rows: r1 - r0,
                            n_cols: self.inner.ncols,
                        },
                    );
                    r0 = r1;
                }
            }
        }

        let (nrows, ncols) = (100, 8);
        let mut dense = vec![1.0_f32; nrows * ncols];
        for (i, v) in dense.iter_mut().enumerate() {
            *v = (i % 7) as f32;
        }
        let m = crate::CsrMatrix::from_dense_row_major(&dense, nrows, ncols);
        let src = Clamped {
            inner: m.clone(),
            cap: 10, // consumer will ask for 50-row blocks; we emit 10
            seen: Mutex::new(Vec::new()),
        };

        let stats = gene_mean_var_blocks(&src, false);
        let want = crate::preprocess::gene_mean_var(&m, false);
        for j in 0..ncols {
            assert!(
                (stats.means[j] - want.means[j]).abs() < 1e-12,
                "gene {j}: {} vs {}",
                stats.means[j],
                want.means[j]
            );
        }
        assert_eq!(src.seen.into_inner().unwrap().len(), 10);
    }
}
