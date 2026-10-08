//! `CompressedCsr`: a ~3-byte-per-nonzero in-RAM CSR for raw count
//! matrices.
//!
//! Gene indices are stored **delta-encoded `u16`** (rows sorted by column;
//! first index absolute, subsequent gaps minus one — valid whenever
//! `n_cols ≤ 65 536`, which every scRNA matrix satisfies). Values use the
//! narrowest exact representation: **`u8`** when every stored value is an
//! integer in `0..=255` (> 99.9 % of raw UMI matrices), **`u16`** up to
//! `65 535`, else **`f32`**. `indptr` is `u64` — nnz can exceed 2³² at
//! 10M cells. Bytes per nonzero: 3 (u8), 4 (u16), 6 (f32) plus one `u64`
//! per row.
//!
//! Decoding a block is a prefix sum over deltas plus a widening of values
//! — trivially vectorisable, ~1 GB/s per core. Implements [`RowBlocks`],
//! so every kernel (QC, gene stats, sparse products, Gram) runs on the
//! compressed form directly; the pipeline materialises only the HVG-subset
//! matrix (2000 of 30k genes) as f32 CSR, keeping the peak at
//! raw-compressed + HVG-float.

#![allow(
    clippy::len_without_is_empty,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use crate::matrix::CsrMatrix;
use crate::rowblocks::RowBlocks;
use crate::rsvd::CsrRef;
use rayon::prelude::*;

/// How a `CompressedCsr` stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// Integers in `0..=255`.
    U8,
    /// Integers in `0..=65_535`.
    U16,
    /// Arbitrary floats (fallback).
    F32,
}

/// A CSR matrix with delta-encoded `u16` column indices and narrowed
/// values. See the module docs for the layout.
#[derive(Debug, Clone)]
pub struct CompressedCsr {
    n_rows: usize,
    n_cols: usize,
    /// Row pointers, `n_rows + 1` entries, over the `deltas`/values arrays.
    indptr: Vec<u64>,
    /// Per-row delta-encoded column indices (first index absolute, then
    /// gaps minus one).
    deltas: Vec<u16>,
    /// Exactly one of the three value arrays is populated, per `kind`.
    values_u8: Vec<u8>,
    values_u16: Vec<u16>,
    values_f32: Vec<f32>,
    kind: ValueKind,
}

impl CompressedCsr {
    /// Compress a CSR matrix. Returns `None` when compression is not
    /// possible: `n_cols > 65 536` (indices would not fit `u16` deltas) or
    /// some row is not sorted by column index (delta encoding requires
    /// order). Matrices from our constructors and the readers are always
    /// sorted.
    #[must_use]
    pub fn from_csr(m: &CsrMatrix) -> Option<Self> {
        if m.ncols > u16::MAX as usize + 1 {
            return None;
        }
        // Value kind: the narrowest exact representation.
        let mut kind = ValueKind::U8;
        for &v in &m.data {
            let f = f64::from(v);
            if !(f.fract() == 0.0 && (0.0..=255.0).contains(&f)) {
                kind = if f.fract() == 0.0 && (0.0..=65_535.0).contains(&f) {
                    ValueKind::U16
                } else {
                    ValueKind::F32
                };
                break;
            }
        }

        let nnz = m.data.len();
        let mut deltas = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(m.nrows + 1);
        indptr.push(0_u64);
        for r in 0..m.nrows {
            let (s, e) = (m.indptr[r] as usize, m.indptr[r + 1] as usize);
            let mut prev: Option<u32> = None;
            for t in s..e {
                let idx = m.indices[t];
                match prev {
                    None => deltas.push(idx as u16),
                    Some(p) => {
                        if idx <= p {
                            return None; // unsorted or duplicated
                        }
                        deltas.push((idx - p - 1) as u16);
                    }
                }
                prev = Some(idx);
            }
            indptr.push(deltas.len() as u64);
        }

        let (vals8, vals16, vals32) = match kind {
            ValueKind::U8 => (
                m.data.iter().map(|&v| v as u8).collect(),
                Vec::new(),
                Vec::new(),
            ),
            ValueKind::U16 => (
                Vec::new(),
                m.data.iter().map(|&v| v as u16).collect(),
                Vec::new(),
            ),
            ValueKind::F32 => (Vec::new(), Vec::new(), m.data.clone()),
        };

        Some(Self {
            n_rows: m.nrows,
            n_cols: m.ncols,
            indptr,
            deltas,
            values_u8: vals8,
            values_u16: vals16,
            values_f32: vals32,
            kind,
        })
    }

    /// Decode to an f32 `CsrMatrix` (exact round trip: values are stored
    /// exactly; indices reconstruct exactly).
    ///
    /// # Panics
    /// Panics if the stored arrays are internally inconsistent (impossible
    /// for a matrix produced by [`Self::from_csr`]).
    #[must_use]
    pub fn to_csr(&self) -> CsrMatrix {
        let nnz = self.deltas.len();
        let mut indices = vec![0_u32; nnz];
        let mut data = vec![0.0_f32; nnz];
        for r in 0..self.n_rows {
            let (s, e) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            let mut idx: u32 = 0;
            for t in s..e {
                idx = if t == s {
                    u32::from(self.deltas[t])
                } else {
                    idx + u32::from(self.deltas[t]) + 1
                };
                indices[t] = idx;
            }
            match self.kind {
                ValueKind::U8 => {
                    for (d, &v) in data[s..e].iter_mut().zip(&self.values_u8[s..e]) {
                        *d = f32::from(v);
                    }
                }
                ValueKind::U16 => {
                    for (d, &v) in data[s..e].iter_mut().zip(&self.values_u16[s..e]) {
                        *d = f32::from(v);
                    }
                }
                ValueKind::F32 => {
                    data[s..e].copy_from_slice(&self.values_f32[s..e]);
                }
            }
        }
        let indptr: Vec<u32> = self.indptr.iter().map(|&p| p as u32).collect();
        CsrMatrix::new(indptr, indices, data, self.n_rows, self.n_cols)
            .expect("compressed matrix decodes to a consistent CSR")
    }

    /// Number of stored values.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.deltas.len()
    }

    /// Number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The value representation in use.
    #[must_use]
    pub fn kind(&self) -> ValueKind {
        self.kind
    }

    /// Heap bytes of the stored arrays (deltas + values + indptr), for the
    /// bytes-per-nonzero accounting.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let values = match self.kind {
            ValueKind::U8 => self.values_u8.capacity(),
            ValueKind::U16 => self.values_u16.capacity() * 2,
            ValueKind::F32 => self.values_f32.capacity() * 4,
        };
        self.deltas.capacity() * 2 + values + self.indptr.capacity() * 8
    }

    /// Values widened to f32 for rows `r0..r1` (helper for the block
    /// decoder).
    fn widen(&self, s: usize, e: usize, out: &mut Vec<f32>) {
        out.clear();
        match self.kind {
            ValueKind::U8 => out.extend(self.values_u8[s..e].iter().map(|&v| f32::from(v))),
            ValueKind::U16 => out.extend(self.values_u16[s..e].iter().map(|&v| f32::from(v))),
            ValueKind::F32 => out.extend_from_slice(&self.values_f32[s..e]),
        }
    }
}

impl RowBlocks for CompressedCsr {
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
            // Decode the block: prefix-sum deltas into u32 indices, widen
            // values into f32, build a rebased indptr.
            let mut indices = Vec::with_capacity(e - s);
            let mut block_indptr = Vec::with_capacity(r1 - r0 + 1);
            for r in r0..r1 {
                let (rs, re) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
                let mut idx: u32 = 0;
                for t in rs..re {
                    idx = if t == rs {
                        u32::from(self.deltas[t])
                    } else {
                        idx + u32::from(self.deltas[t]) + 1
                    };
                    indices.push(idx);
                }
                block_indptr.push((rs - s) as u32);
            }
            block_indptr.push((e - s) as u32);
            let mut values = Vec::with_capacity(e - s);
            self.widen(s, e, &mut values);
            let block = CsrRef {
                data: &values,
                indices: &indices,
                indptr: &block_indptr,
                n_rows: r1 - r0,
                n_cols: self.n_cols,
            };
            f(r0, block);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess;

    fn xorshift_matrix(
        nrows: usize,
        ncols: usize,
        seed: u64,
        value_of: impl Fn(u64) -> f32,
    ) -> CsrMatrix {
        let mut x = seed;
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for _ in 0..=(x % 5) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let j = (x as usize) % ncols;
                dense[i * ncols + j] = value_of(x);
            }
        }
        CsrMatrix::from_dense_row_major(&dense, nrows, ncols)
    }

    #[test]
    fn round_trip_u8_counts() {
        let m = xorshift_matrix(5_000, 300, 7, |x| ((x % 233) + 1) as f32);
        let c = CompressedCsr::from_csr(&m).expect("compressible");
        assert_eq!(c.kind(), ValueKind::U8);
        let back = c.to_csr();
        assert_eq!(back.indptr, m.indptr);
        assert_eq!(back.indices, m.indices);
        assert_eq!(back.data, m.data);
    }

    #[test]
    fn round_trip_u16_counts() {
        // Some values above 255 → u16, still all integers.
        let m = xorshift_matrix(5_000, 300, 11, |x| ((x % 60_000) + 1) as f32);
        let c = CompressedCsr::from_csr(&m).expect("compressible");
        assert_eq!(c.kind(), ValueKind::U16);
        let back = c.to_csr();
        assert_eq!(back.indptr, m.indptr);
        assert_eq!(back.indices, m.indices);
        assert_eq!(back.data, m.data);
    }

    #[test]
    fn round_trip_f32_floats() {
        let m = xorshift_matrix(5_000, 300, 13, |x| (x % 1_000) as f32 * 0.037);
        let c = CompressedCsr::from_csr(&m).expect("compressible");
        assert_eq!(c.kind(), ValueKind::F32);
        let back = c.to_csr();
        assert_eq!(back.indptr, m.indptr);
        assert_eq!(back.indices, m.indices);
        for (x, y) in back.data.iter().zip(&m.data) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "f32 values must round-trip exactly"
            );
        }
    }

    #[test]
    fn u8_memory_is_three_bytes_per_nonzero() {
        // ~150 nnz per row so the u64 indptr amortizes below 0.06 B/nnz,
        // as in real raw scRNA matrices (~1k nnz/row).
        let mut x = 17_u64;
        let mut dense = vec![0.0_f32; 20_000 * 400];
        for i in 0..20_000 {
            for j in 0..400 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if x % 100 < 38 {
                    dense[i * 400 + j] = ((x % 200) + 1) as f32;
                }
            }
        }
        let m = CsrMatrix::from_dense_row_major(&dense, 20_000, 400);
        let nnz = m.data.len();
        assert!(nnz > 20_000 * 400 / 10, "fixture too sparse: {nnz}");
        let c = CompressedCsr::from_csr(&m).expect("compressible");
        let per_nnz = c.heap_bytes() as f64 / nnz as f64;
        assert!(
            per_nnz <= 3.1,
            "heap bytes per nonzero {per_nnz:.3} exceeds 3.1"
        );
        assert!(per_nnz >= 2.9, "suspiciously small: {per_nnz:.3}");
    }

    #[test]
    fn rejects_unsorted_rows_and_wide_matrices() {
        // n_cols beyond u16 deltas → None.
        let mut dense = vec![0.0_f32; 70_000];
        dense[0] = 1.0;
        let wide = CsrMatrix::from_dense_row_major(&dense, 1, 70_000);
        assert!(CompressedCsr::from_csr(&wide).is_none());
        // Hand-built unsorted row → None.
        let mut m = CsrMatrix::from_dense_row_major(&[1.0, 0.0, 2.0, 0.0], 1, 4);
        m.indices.swap(0, 1); // [2, 0] unsorted
        assert!(CompressedCsr::from_csr(&m).is_none());
    }

    #[test]
    fn kernels_on_compressed_equal_kernels_on_csr() {
        use crate::rsvd::{centered_sum_squares_blocks, column_means_blocks};
        let m = xorshift_matrix(9_000, 47, 19, |x| ((x % 250) + 1) as f32);
        let c = CompressedCsr::from_csr(&m).expect("compressible");

        let means_c = column_means_blocks(&c);
        let means_m = column_means_blocks(&m);
        for (x, y) in means_c.iter().zip(&means_m) {
            assert_eq!(x.to_bits(), y.to_bits());
        }

        let ss_c = centered_sum_squares_blocks(&c, None);
        let ss_m = centered_sum_squares_blocks(&m, None);
        assert_eq!(ss_c.to_bits(), ss_m.to_bits());

        let qc_c = preprocess::qc_metrics(&c.to_csr(), None);
        let qc_m = preprocess::qc_metrics(&m, None);
        assert_eq!(qc_c.n_genes, qc_m.n_genes);
        for (x, y) in qc_c.n_counts.iter().zip(&qc_m.n_counts) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
