//! Randomized SVD (Halko–Martinsson–Tropp) for sparse and dense matrices.
//!
//! Computes the top-`k` singular triplets in O(nnz · l) time instead of the
//! O(n · min(n,m)²) of a full SVD — the right algorithm for PCA when only the
//! top ~50 components matter.
//!
//! Algorithm (with `l = k + oversample`):
//! 1. `Y = A · Ω`, `Ω` Gaussian `(n_cols × l)`.
//! 2. Power iterations: `Q = qr(Y)`, `Z = qr(Aᵀ Q)`, `Y = A · Z`.
//! 3. `Q = qr(Y)`; `Bᵀ = Aᵀ Q` (`n_cols × l`).
//! 4. Thin SVD of the small `Bᵀ = P S Wᵀ`; then `U = Q · W`, singular values `S`.
//!
//! ## Implicit centering
//!
//! PCA needs `A − 1μᵀ` (columns centered) but centering a sparse matrix
//! densifies it. Instead both products absorb the correction algebraically:
//! `(A − 1μᵀ) B = A B − 1 (μᵀ B)` and `(A − 1μᵀ)ᵀ Q = Aᵀ Q − μ (1ᵀ Q)`.
//! Pass the column means as [`RsvdOptions::center`] and the matrix is never
//! touched.
//!
//! ## Kernels
//!
//! - [`spmm`] (`A · B`) is parallel over **rows**: each row of the CSR is read
//!   exactly once and produces one dense row of the output. This is the fix
//!   for the earlier column-parallel version, which streamed the whole matrix
//!   once per column of `B`.
//! - [`spmm_t`] (`Aᵀ · Q`) accumulates into per-chunk `n_cols × l` partials
//!   over a **fixed** row chunking and merges them in chunk order, so the
//!   result is bit-identical regardless of thread scheduling. Float addition
//!   is not associative; a fold/reduce would not be reproducible.
//!
//! Everything works on borrowed slices ([`CsrRef`]), so memory-mapped
//! matrices use the same code without a copy.

#![allow(clippy::many_single_char_names, clippy::doc_markdown)]

use crate::matrix::{CsrMatrix, DenseMatrix};
use crate::rowblocks::{BlockFold, RowBlocks};
use crate::simd;
use faer::{
    linalg::solvers::{Qr, Svd},
    Mat, MatRef,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Mutex;

/// A borrowed CSR triple — an in-memory [`CsrMatrix`] or memory-mapped files.
#[derive(Clone, Copy)]
pub struct CsrRef<'a> {
    /// Stored values.
    pub data: &'a [f32],
    /// Column indices.
    pub indices: &'a [u32],
    /// Row pointers (`n_rows + 1`).
    pub indptr: &'a [u32],
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

impl<'a> From<&'a CsrMatrix> for CsrRef<'a> {
    fn from(m: &'a CsrMatrix) -> Self {
        Self {
            data: &m.data,
            indices: &m.indices,
            indptr: &m.indptr,
            n_rows: m.nrows,
            n_cols: m.ncols,
        }
    }
}

/// Options for [`rsvd_sparse`].
#[derive(Debug, Clone, Copy)]
pub struct RsvdOptions<'a> {
    /// Number of singular triplets to return.
    pub k: usize,
    /// Extra sketch columns beyond `k` (Halko et al. recommend 5–10).
    pub oversample: usize,
    /// Krylov depth `q` (blocks beyond the sketch); 2–4 typical. Each
    /// level costs two passes and `l` more basis columns.
    pub n_power_iters: usize,
    /// Seed for the Gaussian sketch.
    pub seed: u64,
    /// Column means for implicit centering (`None` = uncentered / TruncatedSVD).
    pub center: Option<&'a [f32]>,
}

/// Result of a randomized SVD: the top-`k` singular triplets.
pub struct RsvdResult {
    /// Left singular vectors `U` (n_rows × k), row-major.
    pub u: Vec<f32>,
    /// Singular values `S` (k), descending.
    pub s: Vec<f32>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// Number of components returned.
    pub k: usize,
}

// ── deterministic row chunking ───────────────────────────────────────────

/// Rows per chunk for reductions over rows: at most ~32 chunks so the
/// per-chunk partials stay small, never fewer than 32k rows per chunk.
/// Depends only on `n_rows`, which is what makes the merge order fixed.
pub(crate) fn chunk_rows(n_rows: usize) -> usize {
    n_rows.div_ceil(32).max(32_768)
}

/// Column sums of a row-major `(n_rows × l)` matrix, accumulated in `f64`
/// over fixed row chunks and merged in order.
fn column_sums_rowmajor(x: &[f32], n_rows: usize, l: usize) -> Vec<f64> {
    let chunk = chunk_rows(n_rows);
    let partials: Vec<Vec<f64>> = (0..n_rows.div_ceil(chunk))
        .into_par_iter()
        .map(|c| {
            let mut acc = vec![0.0_f64; l];
            for i in c * chunk..((c + 1) * chunk).min(n_rows) {
                for (a, &v) in acc.iter_mut().zip(&x[i * l..(i + 1) * l]) {
                    *a += f64::from(v);
                }
            }
            acc
        })
        .collect();
    let mut out = vec![0.0_f64; l];
    for p in &partials {
        for (o, &v) in out.iter_mut().zip(p) {
            *o += v;
        }
    }
    out
}

/// Per-column means of a row-blocks source (zeros included), `f64`
/// accumulation, fixed reduction order — bit-identical to the previous
/// chunked implementation for any scheduling.
#[must_use]
pub fn column_means_blocks(src: &dyn RowBlocks) -> Vec<f32> {
    let chunk = chunk_rows(src.n_rows());
    let fold: BlockFold<Vec<f64>> = BlockFold::new(|acc, p| {
        for (a, &v) in acc.iter_mut().zip(&p) {
            *a += v;
        }
    });
    src.for_each_block(chunk, &|off, b| {
        let mut acc = vec![0.0_f64; b.n_cols];
        for i in 0..b.n_rows {
            let (s, e) = (b.indptr[i] as usize, b.indptr[i + 1] as usize);
            for (&j, &v) in b.indices[s..e].iter().zip(&b.data[s..e]) {
                acc[j as usize] += f64::from(v);
            }
        }
        fold.store(off, acc);
    });
    let n = src.n_rows().max(1) as f64;
    fold.finish(vec![0.0_f64; src.n_cols()])
        .into_iter()
        .map(|s| (s / n) as f32)
        .collect()
}

/// Per-column means of a CSR matrix ([`column_means_blocks`] on a borrowed
/// view).
#[must_use]
pub fn column_means(a: CsrRef<'_>) -> Vec<f32> {
    column_means_blocks(&a)
}

/// `‖A − 1μᵀ‖²_F = Σ a²_ij − n Σ μ²_j`: the total variance (times `n`) that
/// PCA's variance ratio is measured against. `f64`, fixed order.
#[must_use]
pub fn centered_sum_squares_blocks(src: &dyn RowBlocks, means: Option<&[f32]>) -> f64 {
    let chunk = chunk_rows(src.n_rows());
    let fold: BlockFold<f64> = BlockFold::new(|acc, p| *acc += p);
    src.for_each_block(chunk, &|off, b| {
        let s = b
            .data
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>();
        fold.store(off, s);
    });
    let total = fold.finish(0.0);
    match means {
        Some(mu) => {
            let mu_sq: f64 = mu.iter().map(|&m| f64::from(m) * f64::from(m)).sum();
            total - src.n_rows() as f64 * mu_sq
        }
        None => total,
    }
}

/// [`centered_sum_squares_blocks`] on a borrowed CSR view.
#[must_use]
pub fn centered_sum_squares(a: CsrRef<'_>, means: Option<&[f32]>) -> f64 {
    centered_sum_squares_blocks(&a, means)
}

// ── sparse kernels ───────────────────────────────────────────────────────

/// `(A − 1μᵀ) · B` for row-major `B` (`n_cols × l`); returns row-major
/// `(n_rows × l)`. Each block computes its rows independently (per-row
/// accumulation unchanged by blocking); blocks copy into the shared output
/// under a lock — few, large copies.
///
/// # Panics
/// Panics if `b.len() != n_cols * l`.
#[must_use]
pub fn spmm_blocks(src: &dyn RowBlocks, b: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.n_rows() * l];
    spmm_blocks_into(src, b, l, center, &mut out);
    out
}

/// [`spmm_blocks`] written into `out` (`n_rows × l`, row-major), which may be
/// a memory-mapped file: nothing the size of the result is allocated. Every
/// row is written by the block that holds it, so `out` need not be cleared
/// first, except that a block whose read fails (the source skips it) leaves
/// its rows as they were, zeros in a fresh buffer.
///
/// # Panics
/// Panics if `b.len() != n_cols * l` or `out.len() != n_rows * l`.
pub fn spmm_blocks_into(
    src: &dyn RowBlocks,
    b: &[f32],
    l: usize,
    center: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_rows = src.n_rows();
    let n_cols = src.n_cols();
    assert_eq!(b.len(), n_cols * l, "spmm: B must be n_cols × l");
    assert_eq!(out.len(), n_rows * l, "spmm: the output must be n_rows × l");
    // μᵀB — small (l), sequential, f64.
    let mu_b: Option<Vec<f32>> = center.map(|mu| {
        let mut acc = vec![0.0_f64; l];
        for (j, &m) in mu.iter().enumerate() {
            let m = f64::from(m);
            for (a, &v) in acc.iter_mut().zip(&b[j * l..(j + 1) * l]) {
                *a += m * f64::from(v);
            }
        }
        acc.into_iter().map(|v| v as f32).collect()
    });

    let out: Mutex<&mut [f32]> = Mutex::new(out);
    src.for_each_block(chunk_rows(n_rows), &|off, blk| {
        let mut rows = vec![0.0_f32; blk.n_rows * l];
        for i in 0..blk.n_rows {
            let (s, e) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            let acc = &mut rows[i * l..(i + 1) * l];
            for (&j, &v) in blk.indices[s..e].iter().zip(&blk.data[s..e]) {
                let j = j as usize;
                simd::axpy(acc, v, &b[j * l..(j + 1) * l]);
            }
            if let Some(mb) = &mu_b {
                for (x, &m) in acc.iter_mut().zip(mb) {
                    *x -= m;
                }
            }
        }
        let mut out = out.lock().expect("spmm output poisoned");
        out[off * l..(off + blk.n_rows) * l].copy_from_slice(&rows);
    });
}

/// [`spmm_blocks`] on a borrowed CSR view.
#[must_use]
pub fn spmm(a: CsrRef<'_>, b: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
    spmm_blocks(&a, b, l, center)
}

/// `(A − 1μᵀ)ᵀ · Q` for row-major `Q` (`n_rows × l`); returns row-major
/// `(n_cols × l)`. Bit-reproducible: per-block partials over a fixed row
/// chunking, folded in block order.
///
/// # Panics
/// Panics if `q.len() != n_rows * l`.
#[must_use]
pub fn spmm_t_blocks(src: &dyn RowBlocks, q: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
    let n_rows = src.n_rows();
    let n_cols = src.n_cols();
    assert_eq!(q.len(), n_rows * l, "spmm_t: Q must be n_rows × l");
    let chunk = chunk_rows(n_rows);
    let fold: BlockFold<Vec<f32>> = BlockFold::new(|acc, p| {
        for (a, &v) in acc.iter_mut().zip(&p) {
            *a += v;
        }
    });
    src.for_each_block(chunk, &|off, blk| {
        let mut acc = vec![0.0_f32; n_cols * l];
        for i in 0..blk.n_rows {
            let (s, e) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            let qi = &q[(off + i) * l..(off + i + 1) * l];
            for (&j, &v) in blk.indices[s..e].iter().zip(&blk.data[s..e]) {
                let j = j as usize;
                simd::axpy(&mut acc[j * l..(j + 1) * l], v, qi);
            }
        }
        fold.store(off, acc);
    });

    let mut out = fold.finish(vec![0.0_f32; n_cols * l]);

    if let Some(mu) = center {
        // − μ (1ᵀQ)
        let q_sums = column_sums_rowmajor(q, n_rows, l);
        out.par_chunks_mut(l).enumerate().for_each(|(j, row)| {
            let m = f64::from(mu[j]);
            for (x, &s) in row.iter_mut().zip(&q_sums) {
                *x -= (m * s) as f32;
            }
        });
    }
    out
}

/// [`spmm_t_blocks`] on a borrowed CSR view.
#[must_use]
pub fn spmm_t(a: CsrRef<'_>, q: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
    spmm_t_blocks(&a, q, l, center)
}

// ── Gram matrix (exact PCA path) ─────────────────────────────────────────

/// `AᵀA` as a dense row-major `(n_cols × n_cols)` `f64` matrix, from one pass
/// over the rows (each row contributes the outer product of its nonzeros).
/// Accumulated in `f64` per fixed row chunk and folded in chunk order, so the
/// result is bit-identical run to run; the fold's watermark drops each
/// partial as soon as it merges, bounding live partials to the source's
/// concurrency (the old implementation enforced the same bound with explicit
/// batches of ≤ ~256 MB).
///
/// This is the exact-PCA route: after HVG selection `n_cols` is ~2000, so the
/// Gram matrix is 32 MB and its eigendecomposition is cheap — no randomised
/// approximation of the flat tail of the spectrum is needed at all.
#[must_use]
pub fn gram_matrix_blocks(src: &dyn RowBlocks) -> Vec<f64> {
    let m = src.n_cols();
    let chunk = chunk_rows(src.n_rows());
    let fold: BlockFold<Vec<f64>> = BlockFold::new(|acc, p| {
        acc.par_iter_mut()
            .zip(p.par_iter())
            .for_each(|(x, &y)| *x += y);
    });
    src.for_each_block(chunk, &|off, blk| {
        let mut acc = vec![0.0_f64; m * m];
        for i in 0..blk.n_rows {
            let (s, e) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            let cols = &blk.indices[s..e];
            let vals = &blk.data[s..e];
            for (p, (&ca, &va)) in cols.iter().zip(vals).enumerate() {
                let (ca, va) = (ca as usize, f64::from(va));
                for (&cb, &vb) in cols[p..].iter().zip(&vals[p..]) {
                    let cb = cb as usize;
                    // Upper triangle only; symmetrised at the end.
                    let (lo, hi) = if ca <= cb { (ca, cb) } else { (cb, ca) };
                    acc[lo * m + hi] += va * f64::from(vb);
                }
            }
        }
        fold.store(off, acc);
    });
    let mut c = fold.finish(vec![0.0_f64; m * m]);
    for i in 0..m {
        for j in 0..i {
            c[i * m + j] = c[j * m + i];
        }
    }
    c
}

/// [`gram_matrix_blocks`] on a borrowed CSR view.
#[must_use]
pub fn gram_matrix(a: CsrRef<'_>) -> Vec<f64> {
    gram_matrix_blocks(&a)
}

// ── dense helpers (row-major ⇄ faer) ─────────────────────────────────────

/// Orthonormal basis of the columns of a row-major `(n × l)` matrix, as
/// row-major `(n × l)`.
fn qr_q_rowmajor(y: &[f32], n: usize, l: usize) -> Vec<f32> {
    let y_ref = MatRef::from_row_major_slice(y, n, l);
    let qr = Qr::new(y_ref);
    let q = qr.compute_thin_Q();
    let mut out = vec![0.0_f32; n * l];
    out.par_chunks_mut(l).enumerate().for_each(|(i, row)| {
        for (j, x) in row.iter_mut().enumerate() {
            *x = q[(i, j)];
        }
    });
    out
}

/// Standard-normal `(rows × cols)` row-major matrix from a seeded ChaCha
/// stream (Box–Muller), so the sketch is reproducible across platforms.
///
/// Public because every random projection in the workspace must use the same
/// one: a *uniform* `[0, 1)` projection is dominated by the mean direction
/// and is not a valid sketch.
#[must_use]
pub fn gaussian_rowmajor(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = rows * cols;
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Flip each column of a row-major `(n × k)` matrix so its largest-magnitude
/// entry is positive. SVD signs are otherwise arbitrary.
pub(crate) fn canonicalize_signs(u: &mut [f32], n_rows: usize, k: usize) {
    for j in 0..k {
        let mut max_abs = 0.0_f32;
        let mut max_idx = 0_usize;
        for i in 0..n_rows {
            let a = u[i * k + j].abs();
            if a > max_abs {
                max_abs = a;
                max_idx = i;
            }
        }
        if u[max_idx * k + j] < 0.0 {
            for i in 0..n_rows {
                u[i * k + j] = -u[i * k + j];
            }
        }
    }
}

// ── the algorithm ────────────────────────────────────────────────────────

/// Randomized SVD of a CSR matrix (optionally implicitly centered) by
/// **block Krylov iteration** (Musco & Musco 2015). The matrix is only ever
/// read through [`spmm`] / [`spmm_t`].
///
/// Subspace (power) iteration converges like `(σ_{l+1}/σ_k)^(2q+1)`, which
/// is fine for a few dominant components but hopeless for the flat tail of a
/// real scRNA spectrum (PCs 15–50 differ by a few percent). Keeping every
/// intermediate block `Q_0 … Q_q` and doing the Rayleigh–Ritz projection on
/// the whole Krylov basis costs the same number of passes but converges
/// ∝ 1/√ε instead of 1/ε — the difference between a 79° and a ~1° error on
/// PC 50 at equal cost. Memory is `n_rows × l·(q+1)` floats for the basis.
///
/// # Panics
/// Panics if `center` is given with the wrong length, or if the small SVD
/// fails to converge (does not happen for finite input).
#[must_use]
pub fn rsvd_sparse(a: CsrRef<'_>, opts: &RsvdOptions<'_>) -> RsvdResult {
    rsvd_blocks(&a, opts)
}

/// [`rsvd_sparse`] over any [`RowBlocks`] source, so a memory-mapped or
/// streamed matrix takes the identical route (and gives identical numbers)
/// as one in RAM. Every pass over the data goes through
/// [`spmm_blocks`]/[`spmm_t_blocks`]; nothing but the `n × width` basis is
/// held in memory.
///
/// # Panics
/// Panics if `center` is given with the wrong length, or if the small SVD
/// fails to converge (does not happen for finite input).
#[must_use]
pub fn rsvd_blocks(src: &dyn RowBlocks, opts: &RsvdOptions<'_>) -> RsvdResult {
    let (n, m) = (src.n_rows(), src.n_cols());
    if let Some(mu) = opts.center {
        assert_eq!(mu.len(), m, "center must have one mean per column");
    }
    let k = opts.k.min(n.min(m));
    let l = (k + opts.oversample).min(n.min(m)).max(1);
    let depth = opts.n_power_iters;
    let width = (l * (depth + 1)).min(n.min(m));

    // 1. Krylov basis K = [Q_0, Q_1, …, Q_q] with Q_0 = qr(A'Ω) and
    //    Q_i = qr(A' (A'ᵀ Q_{i-1})); each block orthonormalised, row-major.
    let mut krylov = vec![0.0_f32; n * width];
    let store_block = |krylov: &mut [f32], block: &[f32], offset: usize| {
        krylov
            .par_chunks_mut(width)
            .zip(block.par_chunks(l))
            .for_each(|(row, b)| row[offset..offset + l].copy_from_slice(b));
    };
    let omega = gaussian_rowmajor(m, l, opts.seed);
    let mut block = qr_q_rowmajor(&spmm_blocks(src, &omega, l, opts.center), n, l);
    store_block(&mut krylov, &block, 0);
    for d in 1..=depth {
        if d * l >= width {
            break;
        }
        let z = qr_q_rowmajor(&spmm_t_blocks(src, &block, l, opts.center), m, l);
        block = qr_q_rowmajor(&spmm_blocks(src, &z, l, opts.center), n, l);
        store_block(&mut krylov, &block, d * l);
    }

    // 2. Orthonormal basis of the whole Krylov space and Bᵀ = A'ᵀ Q (m × width).
    let q = qr_q_rowmajor(&krylov, n, width);
    drop(krylov);
    let bt = spmm_t_blocks(src, &q, width, opts.center);

    // 3. Bᵀ = P S Wᵀ  ⇒  B = W S Pᵀ, so U_B = W and U = Q · W[:, :k].
    let bt_ref = MatRef::from_row_major_slice(&bt, m, width);
    let svd = Svd::new_thin(bt_ref).expect("SVD of the small sketch converges");
    let w = svd.V(); // (width × width)
    let s_all: Vec<f32> = svd.S().column_vector().iter().copied().collect();

    // W[:, :k] row-major (width × k) for the row-wise product.
    let mut w_rm = vec![0.0_f32; width * k];
    for i in 0..width {
        for j in 0..k {
            w_rm[i * k + j] = w[(i, j)];
        }
    }
    let mut u = vec![0.0_f32; n * k];
    u.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let qi = &q[i * width..(i + 1) * width];
        for (j, x) in row.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for t in 0..width {
                acc += qi[t] * w_rm[t * k + j];
            }
            *x = acc;
        }
    });
    canonicalize_signs(&mut u, n, k);

    RsvdResult {
        u,
        s: s_all.into_iter().take(k).collect(),
        n_rows: n,
        n_cols: m,
        k,
    }
}

/// Randomized SVD of a **dense** row-major matrix via faer's matmul/QR.
/// Used for dense inputs (embeddings, small densified matrices).
///
/// # Panics
/// Panics if the small SVD fails to converge.
#[must_use]
pub fn rsvd_dense(
    a: &DenseMatrix,
    k: usize,
    oversample: usize,
    n_power_iters: usize,
    seed: u64,
) -> RsvdResult {
    let (n_rows, n_cols) = a.shape();
    let k = k.min(n_rows.min(n_cols));
    let l = (k + oversample).min(n_rows.min(n_cols)).max(1);

    let a_faer = MatRef::from_row_major_slice(&a.data, n_rows, n_cols);
    let omega_rm = gaussian_rowmajor(n_cols, l, seed);
    let omega = MatRef::from_row_major_slice(&omega_rm, n_cols, l);

    let mut y: Mat<f32> = a_faer * omega;
    for _ in 0..n_power_iters {
        let qr = Qr::new(y.as_ref());
        let q = qr.compute_thin_Q();
        let atq = a_faer.transpose() * q; // (n_cols × l)
        let zqr = Qr::new(atq.as_ref());
        y = a_faer * zqr.compute_thin_Q();
    }
    let qr = Qr::new(y.as_ref());
    let q = qr.compute_thin_Q(); // (n_rows × l)
    let b = q.transpose() * a_faer; // (l × n_cols)
    let svd = Svd::new_thin(b.as_ref()).expect("SVD of small matrix converges");
    let u_b = svd.U();
    let s: Vec<f32> = svd.S().column_vector().iter().copied().collect();
    let u_full = q * u_b;

    let mut u = vec![0.0_f32; n_rows * k];
    for i in 0..n_rows {
        for j in 0..k {
            u[i * k + j] = u_full[(i, j)];
        }
    }
    canonicalize_signs(&mut u, n_rows, k);

    RsvdResult {
        u,
        s: s.into_iter().take(k).collect(),
        n_rows,
        n_cols,
        k,
    }
}

// ── legacy oracles: the pre-RowBlocks kernel bodies, verbatim, used by the
// bit-identity tests. Do not "modernise" these — they are the reference.
#[cfg(test)]
pub(super) mod legacy {
    use super::*;
    use crate::rsvd::CsrRef;

    pub fn column_means(a: CsrRef<'_>) -> Vec<f32> {
        let chunk = chunk_rows(a.n_rows);
        let partials: Vec<Vec<f64>> = (0..a.n_rows.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let mut acc = vec![0.0_f64; a.n_cols];
                for i in c * chunk..((c + 1) * chunk).min(a.n_rows) {
                    let (s, e) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
                    for (&j, &v) in a.indices[s..e].iter().zip(&a.data[s..e]) {
                        acc[j as usize] += f64::from(v);
                    }
                }
                acc
            })
            .collect();
        let n = a.n_rows.max(1) as f64;
        let mut out = vec![0.0_f64; a.n_cols];
        for p in &partials {
            for (o, &v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out.into_iter().map(|s| (s / n) as f32).collect()
    }

    pub fn centered_sum_squares(a: CsrRef<'_>, means: Option<&[f32]>) -> f64 {
        let chunk = chunk_rows(a.n_rows);
        let partials: Vec<f64> = (0..a.n_rows.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let (s, e) = (
                    a.indptr[c * chunk] as usize,
                    a.indptr[((c + 1) * chunk).min(a.n_rows)] as usize,
                );
                a.data[s..e]
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
            })
            .collect();
        let total: f64 = partials.iter().sum();
        match means {
            Some(mu) => {
                let mu_sq: f64 = mu.iter().map(|&m| f64::from(m) * f64::from(m)).sum();
                total - a.n_rows as f64 * mu_sq
            }
            None => total,
        }
    }

    pub fn spmm(a: CsrRef<'_>, b: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
        assert_eq!(b.len(), a.n_cols * l, "spmm: B must be n_cols × l");
        let mu_b: Option<Vec<f32>> = center.map(|mu| {
            let mut acc = vec![0.0_f64; l];
            for (j, &m) in mu.iter().enumerate() {
                let m = f64::from(m);
                for (a, &v) in acc.iter_mut().zip(&b[j * l..(j + 1) * l]) {
                    *a += m * f64::from(v);
                }
            }
            acc.into_iter().map(|v| v as f32).collect()
        });
        let mut out = vec![0.0_f32; a.n_rows * l];
        out.par_chunks_mut(l).enumerate().for_each(|(i, acc)| {
            let (s, e) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
            for (&j, &v) in a.indices[s..e].iter().zip(&a.data[s..e]) {
                let j = j as usize;
                simd::axpy(acc, v, &b[j * l..(j + 1) * l]);
            }
            if let Some(mb) = &mu_b {
                for (x, &m) in acc.iter_mut().zip(mb) {
                    *x -= m;
                }
            }
        });
        out
    }

    pub fn spmm_t(a: CsrRef<'_>, q: &[f32], l: usize, center: Option<&[f32]>) -> Vec<f32> {
        assert_eq!(q.len(), a.n_rows * l, "spmm_t: Q must be n_rows × l");
        let chunk = chunk_rows(a.n_rows);
        let partials: Vec<Vec<f32>> = (0..a.n_rows.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let mut acc = vec![0.0_f32; a.n_cols * l];
                for i in c * chunk..((c + 1) * chunk).min(a.n_rows) {
                    let (s, e) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
                    let qi = &q[i * l..(i + 1) * l];
                    for (&j, &v) in a.indices[s..e].iter().zip(&a.data[s..e]) {
                        let j = j as usize;
                        simd::axpy(&mut acc[j * l..(j + 1) * l], v, qi);
                    }
                }
                acc
            })
            .collect();
        let mut out = vec![0.0_f32; a.n_cols * l];
        out.par_chunks_mut(l).enumerate().for_each(|(j, row)| {
            for p in &partials {
                for (x, &v) in row.iter_mut().zip(&p[j * l..(j + 1) * l]) {
                    *x += v;
                }
            }
        });
        if let Some(mu) = center {
            let q_sums = super::column_sums_rowmajor(q, a.n_rows, l);
            out.par_chunks_mut(l).enumerate().for_each(|(j, row)| {
                let m = f64::from(mu[j]);
                for (x, &s) in row.iter_mut().zip(&q_sums) {
                    *x -= (m * s) as f32;
                }
            });
        }
        out
    }

    pub fn gram_matrix(a: CsrRef<'_>) -> Vec<f64> {
        let m = a.n_cols;
        let chunk = chunk_rows(a.n_rows);
        let n_chunks = a.n_rows.div_ceil(chunk);
        let in_flight = ((256usize << 20) / (m * m * 8).max(1)).clamp(1, 8);
        let mut c = vec![0.0_f64; m * m];
        let mut batch_start = 0;
        while batch_start < n_chunks {
            let batch_end = (batch_start + in_flight).min(n_chunks);
            let partials: Vec<Vec<f64>> = (batch_start..batch_end)
                .into_par_iter()
                .map(|ch| {
                    let mut acc = vec![0.0_f64; m * m];
                    for i in ch * chunk..((ch + 1) * chunk).min(a.n_rows) {
                        let (s, e) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
                        let cols = &a.indices[s..e];
                        let vals = &a.data[s..e];
                        for (p, (&ca, &va)) in cols.iter().zip(vals).enumerate() {
                            let (ca, va) = (ca as usize, f64::from(va));
                            for (&cb, &vb) in cols[p..].iter().zip(&vals[p..]) {
                                let cb = cb as usize;
                                let (lo, hi) = if ca <= cb { (ca, cb) } else { (cb, ca) };
                                acc[lo * m + hi] += va * f64::from(vb);
                            }
                        }
                    }
                    acc
                })
                .collect();
            for p in &partials {
                c.par_iter_mut()
                    .zip(p.par_iter())
                    .for_each(|(x, &y)| *x += y);
            }
            batch_start = batch_end;
        }
        for i in 0..m {
            for j in 0..i {
                c[i * m + j] = c[j * m + i];
            }
        }
        c
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::legacy;
    use crate::matrix::CsrMatrix;

    /// Deterministic integer-valued CSR (UMI semantics: exact sums in any
    /// order), sized to span several chunk_rows blocks (chunk_rows(40000) =
    /// 32768 → 2 blocks) and several ROW_BLOCKs.
    fn counts_matrix(nrows: usize, ncols: usize, seed: u64) -> CsrMatrix {
        let mut x = seed;
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for _ in 0..=(x % 5) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let j = (x as usize) % ncols;
                dense[i * ncols + j] = ((x % 7) + 1) as f32;
            }
        }
        CsrMatrix::from_dense_row_major(&dense, nrows, ncols)
    }

    #[test]
    fn rowblocks_kernels_are_bit_identical_to_legacy() {
        let m = counts_matrix(40_000, 37, 0x1234_5678);
        let a = CsrRef::from(&m);

        // column_means
        let new_means = super::column_means(a);
        let old_means = legacy::column_means(a);
        for (x, y) in new_means.iter().zip(&old_means) {
            assert_eq!(x.to_bits(), y.to_bits(), "column_means differs");
        }

        // centered_sum_squares (centered and uncentered)
        for means in [None, Some(&new_means[..])] {
            let new_v = super::centered_sum_squares(a, means);
            let old_v = legacy::centered_sum_squares(a, means);
            assert_eq!(new_v.to_bits(), old_v.to_bits(), "centered_sum_squares");
        }

        // spmm / spmm_t against a fixed l × cols sketch, centered and not.
        let l = 3;
        let b: Vec<f32> = (0..37 * l).map(|i| ((i % 9) as f32 - 4.0) * 0.25).collect();
        for center in [None, Some(&new_means[..])] {
            let new_ab = super::spmm(a, &b, l, center);
            let old_ab = legacy::spmm(a, &b, l, center);
            for (x, y) in new_ab.iter().zip(&old_ab) {
                assert_eq!(x.to_bits(), y.to_bits(), "spmm differs");
            }
            let q: Vec<f32> = (0..40_000 * l)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
                .collect();
            let got_at = super::spmm_t(a, &q, l, center);
            let want_at = legacy::spmm_t(a, &q, l, center);
            for (x, y) in got_at.iter().zip(&want_at) {
                assert_eq!(x.to_bits(), y.to_bits(), "spmm_t differs");
            }
        }

        // gram_matrix
        let small = counts_matrix(40_000, 9, 0xdead_beef);
        let g_new = super::gram_matrix(CsrRef::from(&small));
        let g_old = legacy::gram_matrix(CsrRef::from(&small));
        for (x, y) in g_new.iter().zip(&g_old) {
            assert_eq!(x.to_bits(), y.to_bits(), "gram differs");
        }
    }

    #[test]
    fn rowblocks_kernels_deterministic_across_runs() {
        let m = counts_matrix(40_000, 23, 42);
        let a = CsrRef::from(&m);
        let r1 = super::column_means(a);
        let r2 = super::column_means(a);
        for (x, y) in r1.iter().zip(&r2) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let l = 2;
        let q: Vec<f32> = (0..40_000 * l).map(|i| (i % 7) as f32 * 0.5).collect();
        let t1 = super::spmm_t(a, &q, l, None);
        let t2 = super::spmm_t(a, &q, l, None);
        for (x, y) in t1.iter().zip(&t2) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    use super::*;

    /// Random sparse matrix with a strong low-rank signal plus noise, so the
    /// top singular directions are well separated.
    fn low_rank_csr(n: usize, m: usize, seed: u64) -> CsrMatrix {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut dense = vec![0.0_f32; n * m];
        let u1: Vec<f32> = (0..n).map(|_| rng.random::<f32>()).collect();
        let u2: Vec<f32> = (0..n).map(|_| rng.random::<f32>() - 0.5).collect();
        let v1: Vec<f32> = (0..m).map(|_| rng.random::<f32>()).collect();
        let v2: Vec<f32> = (0..m).map(|_| rng.random::<f32>() - 0.5).collect();
        for i in 0..n {
            for j in 0..m {
                if rng.random::<f32>() < 0.15 {
                    let v = 20.0 * u1[i] * v1[j] + 8.0 * u2[i] * v2[j] + rng.random::<f32>();
                    dense[i * m + j] = v.max(0.0) + 1.0;
                }
            }
        }
        CsrMatrix::from_dense_row_major(&dense, n, m)
    }

    fn naive_spmm(d: &DenseMatrix, b: &[f32], l: usize, mu: Option<&[f32]>) -> Vec<f32> {
        let mut out = vec![0.0_f32; d.nrows * l];
        for i in 0..d.nrows {
            for j in 0..d.ncols {
                let x = d.data[i * d.ncols + j] - mu.map_or(0.0, |m| m[j]);
                for t in 0..l {
                    out[i * l + t] += x * b[j * l + t];
                }
            }
        }
        out
    }

    #[test]
    fn spmm_and_spmm_t_match_dense_with_centering() {
        let csr = low_rank_csr(300, 50, 1);
        let d = csr.to_dense();
        let a = CsrRef::from(&csr);
        let mu = column_means(a);
        let l = 7;
        let b = gaussian_rowmajor(50, l, 3);
        for center in [None, Some(mu.as_slice())] {
            let got = spmm(a, &b, l, center);
            let want = naive_spmm(&d, &b, l, center);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-3 * w.abs().max(1.0), "spmm {g} vs {w}");
            }
            // Aᵀ Q via the dense transpose.
            let q = gaussian_rowmajor(300, l, 4);
            let got_t = spmm_t(a, &q, l, center);
            let mut want_t = vec![0.0_f32; 50 * l];
            for i in 0..300 {
                for j in 0..50 {
                    let x = d.data[i * 50 + j] - center.map_or(0.0, |m| m[j]);
                    for t in 0..l {
                        want_t[j * l + t] += x * q[i * l + t];
                    }
                }
            }
            for (g, w) in got_t.iter().zip(&want_t) {
                assert!((g - w).abs() < 1e-2 * w.abs().max(1.0), "spmm_t {g} vs {w}");
            }
        }
    }

    #[test]
    fn column_means_and_sum_squares_match_dense() {
        let csr = low_rank_csr(200, 30, 2);
        let d = csr.to_dense();
        let a = CsrRef::from(&csr);
        let mu = column_means(a);
        let mut ss = 0.0_f64;
        for j in 0..30 {
            let col: Vec<f64> = (0..200).map(|i| f64::from(d.data[i * 30 + j])).collect();
            let mean = col.iter().sum::<f64>() / 200.0;
            assert!((f64::from(mu[j]) - mean).abs() < 1e-5);
            ss += col.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        }
        let got = centered_sum_squares(a, Some(&mu));
        assert!((got - ss).abs() < 1e-3 * ss, "{got} vs {ss}");
    }

    #[test]
    fn implicit_centering_equals_explicit_centering() {
        let csr = low_rank_csr(400, 60, 5);
        let a = CsrRef::from(&csr);
        let mu = column_means(a);
        // Explicitly centered dense copy.
        let mut d = csr.to_dense();
        for i in 0..d.nrows {
            for j in 0..d.ncols {
                d.data[i * d.ncols + j] -= mu[j];
            }
        }
        let k = 3;
        let sparse = rsvd_sparse(
            a,
            &RsvdOptions {
                k,
                oversample: 10,
                n_power_iters: 3,
                seed: 9,
                center: Some(&mu),
            },
        );
        let dense = rsvd_dense(&d, k, 10, 3, 9);
        for j in 0..k {
            let (s1, s2) = (sparse.s[j], dense.s[j]);
            assert!(
                (s1 - s2).abs() < 1e-2 * s2,
                "singular value {j}: {s1} vs {s2}"
            );
        }
        // Leading left singular vector agrees up to sign (signs canonicalised).
        let mut dot = 0.0_f32;
        for i in 0..400 {
            dot += sparse.u[i * k] * dense.u[i * k];
        }
        assert!(dot.abs() > 0.999, "|cos| = {dot}");
    }

    #[test]
    fn rsvd_sparse_is_bit_identical_across_runs() {
        // Large enough to span several row chunks and engage every thread.
        let csr = low_rank_csr(70_000, 40, 7);
        let a = CsrRef::from(&csr);
        let mu = column_means(a);
        let opts = RsvdOptions {
            k: 5,
            oversample: 5,
            n_power_iters: 2,
            seed: 42,
            center: Some(&mu),
        };
        let r1 = rsvd_sparse(a, &opts);
        let r2 = rsvd_sparse(a, &opts);
        assert_eq!(r1.s, r2.s);
        assert_eq!(r1.u, r2.u, "left singular vectors differ between runs");
    }

    #[test]
    fn rsvd_directions_match_exact_svd() {
        // The test that was missing: singular *vectors*, not just values,
        // against an exact SVD. (`Qr::Q_basis` is the Householder basis, not
        // Q; using it gave right singular values with 45-degree-off vectors.)
        let csr = low_rank_csr(500, 80, 11);
        let a = CsrRef::from(&csr);
        let mu = column_means(a);
        let mut d = csr.to_dense();
        for i in 0..d.nrows {
            for j in 0..d.ncols {
                d.data[i * d.ncols + j] -= mu[j];
            }
        }
        let exact = Svd::new_thin(MatRef::from_row_major_slice(&d.data, 500, 80))
            .expect("exact SVD converges");
        let s_exact = exact.S().column_vector();
        let u_exact = exact.U();

        let k = 4;
        let r = rsvd_sparse(
            a,
            &RsvdOptions {
                k,
                oversample: 10,
                n_power_iters: 4,
                seed: 3,
                center: Some(&mu),
            },
        );
        let rd = rsvd_dense(&d, k, 10, 4, 3);
        for (name, res) in [("sparse", &r), ("dense", &rd)] {
            for j in 0..k {
                let (s1, s2) = (res.s[j], s_exact[j]);
                let tol = if j < 2 { 2e-3 } else { 1e-2 }; // noise tail is less separated
                assert!((s1 - s2).abs() < tol * s2, "{name} sigma {j}: {s1} vs {s2}");
                let mut dot = 0.0_f32;
                let mut norm = 0.0_f32;
                for i in 0..500 {
                    dot += res.u[i * k + j] * u_exact[(i, j)];
                    norm += res.u[i * k + j] * res.u[i * k + j];
                }
                assert!((norm - 1.0).abs() < 1e-3, "{name} U column {j} norm {norm}");
                // Only the two dominant programs (weights 20 and 8) have a
                // spectral gap; the rest are noise directions.
                if j < 2 {
                    assert!(
                        dot.abs() > 0.999,
                        "{name} U column {j}: |cos| = {}",
                        dot.abs()
                    );
                }
            }
        }
    }

    #[test]
    fn rsvd_dense_recovers_top_components() {
        let mut data = vec![0.0_f32; 50 * 5];
        for i in 0..50 {
            let t = i as f32;
            data[i * 5] = t;
            data[i * 5 + 1] = t * -2.0 + (i % 3) as f32 * 0.001;
        }
        let d = DenseMatrix {
            data,
            nrows: 50,
            ncols: 5,
        };
        let result = rsvd_dense(&d, 2, 3, 2, 42);
        assert_eq!(result.k, 2);
        assert!(result.s[0] > result.s[1]);
        assert!(result.s[0] > 10.0);
        let again = rsvd_dense(&d, 2, 3, 2, 42);
        assert_eq!(result.u, again.u);
    }
}
