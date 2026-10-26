//! Principal Component Analysis via SVD.
//!
//! The standard scanpy/sklearn approach: center the columns (subtract per-gene
//! mean), compute the truncated SVD, and return `U·diag(S)` as the embedding.
//! Variance ratio is `S²/(n−1)` normalized by total variance.
//!
//! Two paths:
//! - **Dense**: thin SVD via [`faer`] on the densified matrix. Good for small
//!   HVG-subset matrices (`n_features` ≤ a few thousand) and for dense
//!   embeddings.
//! - **Sparse**: randomized SVD. A sparse `X` (`n_obs × n_features`, 90%+
//!   zeros) is multiplied by a small Gaussian random matrix (`n_features ×
//!   oversampled rank`), yielding a dense `n_obs × (rank·oversample)` sketch
//!   whose SVD approximates the top components of `X`. This keeps `X` sparse
//!   throughout the expensive projection step — the key memory/speed win on
//!   real omics.

#![allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use crate::backend::PcaResult;
use crate::matrix::{CsrMatrix, DenseMatrix, Matrix};
use crate::rowblocks::RowBlocks;
use faer::{linalg::solvers::Svd, Mat};
use rayon::prelude::*;

/// Errors raised by PCA.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PcaError {
    /// Requested more components than the rank of the data allows.
    #[error("n_components ({requested}) exceeds min(n_obs, n_features) ({max})")]
    TooManyComponents {
        /// Requested component count.
        requested: usize,
        /// Maximum allowed: `min(n_obs, n_features)`.
        max: usize,
    },
    /// The input matrix has zero rows or columns.
    #[error("input matrix has zero rows ({nrows}) or columns ({ncols})")]
    EmptyInput {
        /// Row count.
        nrows: usize,
        /// Column count.
        ncols: usize,
    },
    /// The underlying SVD failed to converge.
    #[error("SVD failed to converge: {0}")]
    SvdFailed(String),
}

/// PCA options.
#[derive(Debug, Clone)]
pub struct PcaOptions {
    /// Number of principal components to compute.
    pub n_components: usize,
    /// Whether to center columns (subtract per-gene mean) before SVD.
    /// Matches scanpy's `zero_center` (default `true`).
    pub center: bool,
    /// Oversampling parameter for randomized SVD (sparse path). Extra dimensions
    /// beyond `n_components` to improve accuracy. Halko et al. recommend 2–10.
    pub oversample: usize,
    /// Number of power iterations for randomized SVD. 0–2 typical; higher
    /// improves accuracy for slowly-decaying spectra.
    pub n_iter: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for PcaOptions {
    fn default() -> Self {
        Self {
            n_components: 50,
            center: true,
            oversample: 10,
            n_iter: 1,
            seed: 0,
        }
    }
}

/// What a PCA that wrote its embedding into a caller's buffer returns: all of
/// [`PcaResult`] but the embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaSummary {
    /// Variance ratio per component.
    pub variance_ratio: Vec<f32>,
    /// Observations (rows of the embedding).
    pub n_obs: usize,
    /// Components computed.
    pub n_components: usize,
}

impl PcaResult {
    /// Split off the embedding, leaving the summary.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f32>, PcaSummary) {
        (
            self.embedding,
            PcaSummary {
                variance_ratio: self.variance_ratio,
                n_obs: self.n_obs,
                n_components: self.n_components,
            },
        )
    }
}

/// Copy an embedding computed in memory into the caller's buffer (the paths
/// that cannot write their rows in place: tiny matrices, wide randomized SVD).
fn deliver(result: PcaResult, out: &mut [f32]) -> PcaSummary {
    let (emb, summary) = result.into_parts();
    out.copy_from_slice(&emb);
    summary
}

/// Run PCA on a [`Matrix`], dispatching to the optimal algorithm.
///
/// Three-way adaptive dispatch:
/// - **Dense input, few components** (`n_components < min(n,m)/4`): randomized
///   SVD via faer's optimized matmul/QR. Computes only the top-k — dramatically
///   faster than a full thin SVD when you need 50 of 2000 components.
/// - **Dense input, many components**: exact thin SVD.
/// - **Sparse input**: randomized SVD that keeps X sparse throughout the
///   projection step (the memory win for 90%+ sparse omics data).
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD failure.
pub fn pca(matrix: &Matrix, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    match matrix {
        Matrix::Dense(d) => pca_dense_dispatch(d, opts),
        Matrix::Csr(s) => pca_csr(s, opts),
    }
}

/// Randomized SVD is worth it when few components are needed relative to the
/// rank AND the matrix is large enough to amortize the random projection. On
/// small matrices (< ~1000 rows or cols) the exact thin SVD is sub-millisecond
/// and exact — randomized SVD would add error for zero speed benefit.
fn use_rsvd(n_obs: usize, n_features: usize, n_components: usize) -> bool {
    let min_dim = n_obs.min(n_features);
    n_components * 4 < min_dim && min_dim >= 1_000
}

/// Assemble `U·diag(S)` and the variance ratio from an rSVD result.
fn finish_rsvd(r: &crate::rsvd::RsvdResult, total_var: f64) -> PcaResult {
    let (n_obs, k) = (r.n_rows, r.k);
    let mut embedding = vec![0.0_f32; n_obs * k];
    for i in 0..n_obs {
        for j in 0..k {
            embedding[i * k + j] = r.u[i * k + j] * r.s[j];
        }
    }
    let variance_ratio: Vec<f32> = (0..k)
        .map(|j| (f64::from(r.s[j] * r.s[j]) / total_var) as f32)
        .collect();
    PcaResult {
        embedding,
        variance_ratio,
        n_obs,
        n_components: k,
    }
}

fn sum_of_squares(data: &[f32]) -> f64 {
    data.iter()
        .copied()
        .map(|x| f64::from(x) * f64::from(x))
        .sum()
}

/// Dense input: randomized SVD for few components, exact thin SVD otherwise.
fn pca_dense_dispatch(d: &DenseMatrix, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    let (n_obs, n_features) = d.shape();
    validate_dims(n_obs, n_features, opts.n_components)?;
    if !use_rsvd(n_obs, n_features, opts.n_components) {
        return pca_dense(d, opts);
    }
    let centered = if opts.center {
        center_dense(d)
    } else {
        d.clone()
    };
    let total_var = sum_of_squares(&centered.data);
    let r = crate::rsvd::rsvd_dense(
        &centered,
        opts.n_components,
        opts.oversample,
        opts.n_iter,
        opts.seed,
    );
    Ok(finish_rsvd(&r, total_var))
}

/// Above this many features the `n_features²` covariance stops being cheap
/// (f64: 4096² × 8 B = 134 MB, and its eigendecomposition seconds) and PCA
/// falls back to randomized block Krylov SVD.
pub const GRAM_MAX_FEATURES: usize = 4096;

/// PCA on a CSR matrix, taken by reference (no clone at the FFI boundary).
///
/// Three paths, in order of preference:
/// - **Tiny** (`min(n, m) < 1000` or many components): exact thin SVD of the
///   densified matrix.
/// - **`n_features ≤ 4096`** (the normal case after HVG selection): **exact**
///   PCA through the centered Gram matrix `AᵀA − nμμᵀ` — one deterministic
///   pass to build it, a dense symmetric eigendecomposition, one pass to
///   project `A'V`. No randomness; the flat tail of the spectrum is resolved
///   exactly.
/// - **Wider**: randomized block Krylov SVD with implicit centering
///   ([`crate::rsvd`]); accuracy on the tail depends on `oversample`/`n_iter`.
///
/// Nothing on the sparse paths densifies or modifies the matrix. The variance
/// ratio is `λ_j / Σλ` (Gram) or `s_j² / ‖A − 1μᵀ‖²_F` (rSVD) — scanpy's
/// convention either way.
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
pub fn pca_csr(s: &CsrMatrix, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    let (n_obs, n_features) = s.shape();
    validate_dims(n_obs, n_features, opts.n_components)?;
    let mut embedding = vec![0.0_f32; n_obs * opts.n_components];
    let summary = pca_csr_into(s, opts, &mut embedding)?;
    Ok(PcaResult {
        embedding,
        variance_ratio: summary.variance_ratio,
        n_obs: summary.n_obs,
        n_components: summary.n_components,
    })
}

/// [`pca_csr`] with the embedding written into `out` (`n_obs × n_components`,
/// row-major; zeros, or a memory-mapped file): on the exact Gram path nothing
/// the size of the embedding is allocated. The wide randomized path and the
/// tiny exact path compute the embedding in memory and copy it over.
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
///
/// # Panics
/// Panics if `out.len() != n_obs * n_components`.
pub fn pca_csr_into(
    s: &CsrMatrix,
    opts: &PcaOptions,
    out: &mut [f32],
) -> Result<PcaSummary, PcaError> {
    let (n_obs, n_features) = s.shape();
    validate_dims(n_obs, n_features, opts.n_components)?;
    assert_eq!(
        out.len(),
        n_obs * opts.n_components,
        "output must be n_obs × n_components"
    );
    if !use_rsvd(n_obs, n_features, opts.n_components) {
        return Ok(deliver(pca_dense(&s.to_dense(), opts)?, out));
    }
    if n_features <= GRAM_MAX_FEATURES {
        return pca_gram_blocks_into(&crate::rsvd::CsrRef::from(s), opts, out);
    }
    let a = crate::rsvd::CsrRef::from(s);
    let means = opts.center.then(|| crate::rsvd::column_means(a));
    let total_var = crate::rsvd::centered_sum_squares(a, means.as_deref());
    let r = crate::rsvd::rsvd_sparse(
        a,
        &crate::rsvd::RsvdOptions {
            k: opts.n_components,
            oversample: opts.oversample,
            n_power_iters: opts.n_iter,
            seed: opts.seed,
            center: means.as_deref(),
        },
    );
    Ok(deliver(finish_rsvd(&r, total_var), out))
}

/// PCA over any [`RowBlocks`] source — in-RAM, memory-mapped or streamed
/// from disk — so an out-of-core matrix takes the same route as an in-memory
/// one and produces the same numbers.
///
/// Uses the exact Gram path when `n_features <= GRAM_MAX_FEATURES` (the
/// normal case after HVG selection) and randomized block Krylov above it.
/// Both read the source block by block; neither materialises it. There is no
/// dense fallback here: a source small enough for the thin-SVD path does not
/// need to be streamed.
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
pub fn pca_blocks(src: &dyn RowBlocks, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    let (n_obs, n_features) = (src.n_rows(), src.n_cols());
    validate_dims(n_obs, n_features, opts.n_components)?;
    let mut embedding = vec![0.0_f32; n_obs * opts.n_components];
    let summary = pca_blocks_into(src, opts, &mut embedding)?;
    Ok(PcaResult {
        embedding,
        variance_ratio: summary.variance_ratio,
        n_obs: summary.n_obs,
        n_components: summary.n_components,
    })
}

/// [`pca_blocks`] with the embedding written into `out` (`n_obs ×
/// n_components`, row-major; zeros, or a memory-mapped file). On the Gram
/// path each block of rows is projected straight into `out`, so a result of
/// tens of millions of cells is never held in memory.
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
///
/// # Panics
/// Panics if `out.len() != n_obs * n_components`.
pub fn pca_blocks_into(
    src: &dyn RowBlocks,
    opts: &PcaOptions,
    out: &mut [f32],
) -> Result<PcaSummary, PcaError> {
    let (n_obs, n_features) = (src.n_rows(), src.n_cols());
    validate_dims(n_obs, n_features, opts.n_components)?;
    assert_eq!(
        out.len(),
        n_obs * opts.n_components,
        "output must be n_obs × n_components"
    );
    if n_features <= GRAM_MAX_FEATURES {
        return pca_gram_blocks_into(src, opts, out);
    }
    let means = opts.center.then(|| crate::rsvd::column_means_blocks(src));
    let total_var = crate::rsvd::centered_sum_squares_blocks(src, means.as_deref());
    let r = crate::rsvd::rsvd_blocks(
        src,
        &crate::rsvd::RsvdOptions {
            k: opts.n_components,
            oversample: opts.oversample,
            n_power_iters: opts.n_iter,
            seed: opts.seed,
            center: means.as_deref(),
        },
    );
    Ok(deliver(finish_rsvd(&r, total_var), out))
}

/// PCA of the per-gene standardized matrix: scanpy's `pp.scale` (zero
/// centre, divide by the `ddof = 1` standard deviation, zero-variance genes
/// left at 1, clip to `±max_value`) followed by `tl.pca`, without forming
/// the scaled matrix.
///
/// Scaled data are dense (a zero becomes `−μ_j/σ_j`), so they are never
/// written. Instead the clipped standardized matrix is split as
/// `Z = 1bᵀ + S`, where `b_j` is the value every zero takes in gene `j` and
/// `S` is sparse with the pattern of `X` (`S_ij = clip((x_ij − μ_j)/σ_j) − b_j`).
/// A constant shift of each column changes neither a centred covariance nor
/// a centred projection, so PCA of `Z` is exactly the ordinary centred PCA
/// of `S`, which [`pca_blocks`] computes over a view that rewrites the
/// stored values block by block. PCA is always centred here (`opts.center`
/// is ignored), as scanpy's scaling is.
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
pub fn pca_scaled_blocks(
    src: &dyn RowBlocks,
    opts: &PcaOptions,
    max_value: Option<f32>,
) -> Result<PcaResult, PcaError> {
    validate_dims(src.n_rows(), src.n_cols(), opts.n_components)?;
    let mut embedding = vec![0.0_f32; src.n_rows() * opts.n_components];
    let summary = pca_scaled_blocks_into(src, opts, max_value, &mut embedding)?;
    Ok(PcaResult {
        embedding,
        variance_ratio: summary.variance_ratio,
        n_obs: summary.n_obs,
        n_components: summary.n_components,
    })
}

/// [`pca_scaled_blocks`] with the embedding written into `out`
/// (`n_rows × n_components`, row-major).
///
/// # Errors
/// Returns [`PcaError`] on invalid inputs or SVD/eigen failure.
///
/// # Panics
/// Panics if `out.len() != n_rows * n_components`.
pub fn pca_scaled_blocks_into(
    src: &dyn RowBlocks,
    opts: &PcaOptions,
    max_value: Option<f32>,
    out: &mut [f32],
) -> Result<PcaSummary, PcaError> {
    validate_dims(src.n_rows(), src.n_cols(), opts.n_components)?;
    let stats = crate::preprocess::gene_mean_var_blocks(src, false);
    let clip = max_value.map_or(f64::INFINITY, f64::from);
    let mut mean = Vec::with_capacity(src.n_cols());
    let mut inv_sd = Vec::with_capacity(src.n_cols());
    let mut zero = Vec::with_capacity(src.n_cols());
    for (&mu, &var) in stats.means.iter().zip(&stats.variances) {
        let sd = var.sqrt();
        let inv = if sd == 0.0 { 1.0 } else { 1.0 / sd };
        mean.push(mu);
        inv_sd.push(inv);
        zero.push((-mu * inv).clamp(-clip, clip));
    }
    let view = ScaledView {
        inner: src,
        mean,
        inv_sd,
        zero,
        clip,
    };
    let centred = PcaOptions {
        center: true,
        ..opts.clone()
    };
    pca_blocks_into(&view, &centred, out)
}

/// The sparse part `S` of the scaled matrix (see [`pca_scaled_blocks`]):
/// stored value `x` in gene `j` becomes `clip((x − μ_j)/σ_j) − b_j`.
struct ScaledView<'a> {
    inner: &'a dyn RowBlocks,
    mean: Vec<f64>,
    inv_sd: Vec<f64>,
    /// `b_j`, the clipped value of a zero.
    zero: Vec<f64>,
    clip: f64,
}

impl RowBlocks for ScaledView<'_> {
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    fn for_each_block(
        &self,
        rows_per_block: usize,
        f: &(dyn Fn(usize, crate::rsvd::CsrRef<'_>) + Sync),
    ) {
        self.inner.for_each_block(rows_per_block, &|off, blk| {
            let data: Vec<f32> = blk
                .indices
                .iter()
                .zip(blk.data)
                .map(|(&j, &x)| {
                    let j = j as usize;
                    let z = ((f64::from(x) - self.mean[j]) * self.inv_sd[j])
                        .clamp(-self.clip, self.clip);
                    (z - self.zero[j]) as f32
                })
                .collect();
            f(
                off,
                crate::rsvd::CsrRef {
                    data: &data,
                    indices: blk.indices,
                    indptr: blk.indptr,
                    n_rows: blk.n_rows,
                    n_cols: blk.n_cols,
                },
            );
        });
    }
}

/// Exact PCA through the (centered) Gram matrix, over any [`RowBlocks`]
/// source, with the embedding written into `out` (`n × k`, row-major).
///
/// Builds `C = AᵀA − nμμᵀ` in one deterministic pass, takes its symmetric
/// eigendecomposition (the `m × m` matrix is small after gene selection), and
/// projects `A'V` in a second pass whose blocks write their rows straight into
/// `out`. The column signs are then fixed in place.
fn pca_gram_blocks_into(
    src: &dyn RowBlocks,
    opts: &PcaOptions,
    out: &mut [f32],
) -> Result<PcaSummary, PcaError> {
    use crate::rsvd::{column_means_blocks, gram_matrix_blocks, spmm_blocks_into};
    let (n, m, k) = (src.n_rows(), src.n_cols(), opts.n_components);

    let mut c = gram_matrix_blocks(src);
    let means = opts.center.then(|| column_means_blocks(src));
    if let Some(mu) = &means {
        let nf = n as f64;
        c.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
            let mi = f64::from(mu[i]) * nf;
            for (j, x) in row.iter_mut().enumerate() {
                *x -= mi * f64::from(mu[j]);
            }
        });
    }
    let total_var: f64 = (0..m).map(|i| c[i * m + i]).sum();

    let c_ref = faer::MatRef::from_row_major_slice(&c, m, m);
    let eig = c_ref
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| PcaError::SvdFailed(format!("{e:?}")))?;
    let vals = eig.S().column_vector();
    let vecs = eig.U();

    let mut v = vec![0.0_f32; m * k];
    for j in 0..k {
        let col = m - 1 - j;
        for i in 0..m {
            v[i * k + j] = vecs[(i, col)] as f32;
        }
    }
    spmm_blocks_into(src, &v, k, means.as_deref(), out);
    crate::rsvd::canonicalize_signs(out, n, k);

    let variance_ratio: Vec<f32> = (0..k)
        .map(|j| (vals[m - 1 - j].max(0.0) / total_var) as f32)
        .collect();
    Ok(PcaSummary {
        variance_ratio,
        n_obs: n,
        n_components: k,
    })
}

/// Exact PCA through the (centered) Gram matrix. See [`pca_csr`].
#[cfg(test)]
fn pca_gram(s: &CsrMatrix, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    let mut embedding = vec![0.0_f32; s.nrows * opts.n_components];
    let summary = pca_gram_blocks_into(&crate::rsvd::CsrRef::from(s), opts, &mut embedding)?;
    Ok(PcaResult {
        embedding,
        variance_ratio: summary.variance_ratio,
        n_obs: summary.n_obs,
        n_components: summary.n_components,
    })
}

/// Validate input dimensions for PCA.
fn validate_dims(n_obs: usize, n_features: usize, n_components: usize) -> Result<(), PcaError> {
    if n_obs == 0 || n_features == 0 {
        return Err(PcaError::EmptyInput {
            nrows: n_obs,
            ncols: n_features,
        });
    }
    let max_components = n_obs.min(n_features);
    if n_components > max_components {
        return Err(PcaError::TooManyComponents {
            requested: n_components,
            max: max_components,
        });
    }
    Ok(())
}

/// Dense PCA via faer thin SVD.
fn pca_dense(d: &DenseMatrix, opts: &PcaOptions) -> Result<PcaResult, PcaError> {
    let (n_obs, n_features) = d.shape();
    if n_obs == 0 || n_features == 0 {
        return Err(PcaError::EmptyInput {
            nrows: n_obs,
            ncols: n_features,
        });
    }
    let max_components = n_obs.min(n_features);
    if opts.n_components > max_components {
        return Err(PcaError::TooManyComponents {
            requested: opts.n_components,
            max: max_components,
        });
    }

    // Center columns: compute means in f64 for stability, subtract in f32.
    // Then build the faer col-major matrix directly in f32 (half the memory
    // traffic of f64, which is the dominant cost for SVD on wide matrices).
    let mut mat = Mat::<f32>::zeros(n_obs, n_features);
    if opts.center {
        // Parallel column means via rayon.
        let col_means: Vec<f32> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.0_f64;
                for i in 0..n_obs {
                    sum += f64::from(d.data[i * n_features + j]);
                }
                (sum / n_obs as f64) as f32
            })
            .collect();
        // Fill the col-major matrix in parallel by column.
        mat.par_col_iter_mut()
            .zip(col_means.par_iter())
            .enumerate()
            .for_each(|(j, (mut col, &mean))| {
                for i in 0..n_obs {
                    col[i] = d.data[i * n_features + j] - mean;
                }
            });
    } else {
        mat.par_col_iter_mut().enumerate().for_each(|(j, mut col)| {
            for i in 0..n_obs {
                col[i] = d.data[i * n_features + j];
            }
        });
    }

    // THIN SVD: U is (n_obs × k), S is (k), V is (n_features × k),
    // where k = min(n_obs, n_features). Dramatically cheaper than full SVD.
    let svd = Svd::new_thin(mat.as_ref()).map_err(|e| PcaError::SvdFailed(format!("{e:?}")))?;
    let k = opts.n_components;

    // Embedding: U[:, :k] · diag(S[:k])  →  (n_obs × k), row-major f32.
    // Now operating in f32 throughout (singular values are f32).
    let u = svd.U();
    let s = svd.S().column_vector();
    let s_vec: Vec<f32> = s.iter().copied().collect();
    let mut embedding = Vec::with_capacity(n_obs * k);
    for i in 0..n_obs {
        for j in 0..k {
            embedding.push(u[(i, j)] * s_vec[j]);
        }
    }

    // Variance ratio: S² / total_variance, matching scanpy's convention.
    //   - center=true:  total = sum of ALL singular values² (thin SVD gives all
    //     min(n,m); the (n_obs−1) divisor cancels in the ratio).
    //   - center=false: total = sum of squares of raw data entries.
    let total_var: f64 = if opts.center {
        s_vec.iter().copied().map(|sv| f64::from(sv * sv)).sum()
    } else {
        d.data
            .iter()
            .copied()
            .map(|x| f64::from(x) * f64::from(x))
            .sum()
    };
    let variance_ratio: Vec<f32> = (0..k)
        .map(|j| (f64::from(s_vec[j] * s_vec[j]) / total_var) as f32)
        .collect();

    Ok(PcaResult {
        embedding,
        variance_ratio,
        n_obs,
        n_components: k,
    })
}

/// Center a dense matrix's columns (subtract per-gene mean), returning a new
/// owned matrix. Means computed in f64 for stability, subtraction in f32.
fn center_dense(d: &DenseMatrix) -> DenseMatrix {
    let (nrows, ncols) = d.shape();
    let col_means: Vec<f32> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let mut sum = 0.0_f64;
            for i in 0..nrows {
                sum += f64::from(d.data[i * ncols + j]);
            }
            (sum / nrows as f64) as f32
        })
        .collect();
    let mut data = d.data.clone();
    data.par_chunks_exact_mut(ncols).for_each(|row| {
        for (j, x) in row.iter_mut().enumerate() {
            *x -= col_means[j];
        }
    });
    DenseMatrix { data, nrows, ncols }
}

// (rand 0.10's `random::<f64>()` needs no Distribution import.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CsrMatrix;
    use rand::{RngExt, SeedableRng};

    fn rank_two_dense() -> DenseMatrix {
        // Two informative columns + noise. n_obs=50, n_features=5.
        // Columns 0,1 carry the structure; 2,3,4 are tiny noise.
        let mut data = vec![0.0_f32; 50 * 5];
        for i in 0..50 {
            let t = i as f32;
            data[i * 5] = t * 1.0; // col 0: ramp
            data[i * 5 + 1] = t * -2.0 + (i % 3) as f32 * 0.001; // col 1: correlated
            data[i * 5 + 2] = ((i as f32) * 0.7).sin() * 1e-6; // noise
            data[i * 5 + 3] = 0.0;
            data[i * 5 + 4] = 0.0;
        }
        DenseMatrix {
            data,
            nrows: 50,
            ncols: 5,
        }
    }

    #[test]
    fn dense_pca_recovers_low_rank_structure() {
        let d = rank_two_dense();
        let opts = PcaOptions {
            n_components: 2,
            ..Default::default()
        };
        let result = pca_dense(&d, &opts).unwrap();
        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.embedding.len(), 50 * 2);
        // Top 2 components should capture essentially all variance.
        let total: f32 = result.variance_ratio.iter().sum();
        assert!(
            total > 0.999,
            "top 2 should capture ~all variance, got {total}"
        );
        // First component dominates.
        assert!(result.variance_ratio[0] > result.variance_ratio[1]);
    }

    #[test]
    fn dense_pca_rejects_too_many_components() {
        let d = rank_two_dense();
        let opts = PcaOptions {
            n_components: 10, // > min(50, 5) = 5
            ..Default::default()
        };
        let err = pca_dense(&d, &opts).unwrap_err();
        assert_eq!(
            err,
            PcaError::TooManyComponents {
                requested: 10,
                max: 5
            }
        );
    }

    #[test]
    fn dense_pca_rejects_empty() {
        let d = DenseMatrix::zeros(0, 5);
        let err = pca_dense(&d, &PcaOptions::default()).unwrap_err();
        assert!(matches!(err, PcaError::EmptyInput { .. }));
    }

    #[test]
    fn sparse_pca_matches_dense_on_same_data() {
        // Build the same rank-2 data as a CSR (densify then sparsify).
        let d = rank_two_dense();
        let csr = CsrMatrix::from_dense_row_major(&d.data, d.nrows, d.ncols);
        let opts = PcaOptions {
            n_components: 2,
            oversample: 5,
            n_iter: 2,
            seed: 42,
            center: false,
        };
        let sparse_result = pca(&Matrix::Csr(csr), &opts).unwrap();
        let dense_result = pca(&Matrix::Dense(d), &opts).unwrap();

        // Embeddings should have the same scale (singular values match), though
        // signs/components may differ. Compare via total captured variance.
        let sparse_total: f32 = sparse_result.variance_ratio.iter().sum();
        let dense_total: f32 = dense_result.variance_ratio.iter().sum();
        assert!(
            (sparse_total - dense_total).abs() < 0.1,
            "sparse {sparse_total} vs dense {dense_total}"
        );
        // Both should agree the data is ~rank-2.
        assert!(sparse_total > 0.95, "sparse total {sparse_total}");
    }

    /// Scanpy's `pp.scale(max_value)` on a dense copy: centre, divide by the
    /// `ddof = 1` standard deviation (0 → 1), clip to `±max_value`.
    fn scale_dense_reference(dense: &[f32], n: usize, m: usize, max_value: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * m];
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| f64::from(dense[i * m + j])).collect();
            let mu = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1) as f64;
            let sd = if var == 0.0 { 1.0 } else { var.sqrt() };
            let c = f64::from(max_value);
            for i in 0..n {
                out[i * m + j] = ((col[i] - mu) / sd).clamp(-c, c) as f32;
            }
        }
        out
    }

    #[test]
    fn scaled_pca_matches_dense_scale_then_pca() {
        // Sparse counts with a few extreme values (upper clipping), one
        // nearly constant gene with rare zeros (its zeros clip at −max), and
        // one all-zero gene (σ = 0 → 1).
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let (n, m, k) = (500, 30, 5);
        let mut dense = vec![0.0_f32; n * m];
        for i in 0..n {
            for j in 0..m - 2 {
                if rng.random::<f32>() < 0.25 {
                    dense[i * m + j] = (rng.random::<f32>() * 5.0).floor() + 1.0;
                }
            }
            dense[i * m + m - 2] = if i % 250 == 0 {
                0.0
            } else {
                5.0 + rng.random::<f32>() * 0.01
            };
        }
        for i in (0..n).step_by(97) {
            dense[i * m + 3] = 400.0;
        }
        let csr = CsrMatrix::from_dense_row_major(&dense, n, m);
        let opts = PcaOptions {
            n_components: k,
            ..Default::default()
        };
        for max_value in [None, Some(3.0_f32), Some(10.0_f32)] {
            let reference = pca_dense(
                &DenseMatrix::from_row_major(
                    scale_dense_reference(&dense, n, m, max_value.unwrap_or(f32::INFINITY)),
                    n,
                    m,
                )
                .unwrap(),
                &opts,
            )
            .unwrap();
            let scaled = pca_scaled_blocks(&csr, &opts, max_value).unwrap();
            for j in 0..k {
                let (a, b) = (scaled.variance_ratio[j], reference.variance_ratio[j]);
                assert!((a - b).abs() < 1e-4, "{max_value:?} vr {j}: {a} vs {b}");
                let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
                for i in 0..n {
                    let (x, y) = (
                        f64::from(scaled.embedding[i * k + j]),
                        f64::from(reference.embedding[i * k + j]),
                    );
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                }
                let cos = dot.abs() / (na.sqrt() * nb.sqrt());
                assert!(cos > 0.9999, "{max_value:?} component {j}: |cos| = {cos}");
                assert!(
                    (na / nb - 1.0).abs() < 1e-3,
                    "{max_value:?} component {j} scale"
                );
            }
        }
    }

    #[test]
    fn gram_pca_equals_exact_thin_svd() {
        // 600 × 40 random sparse counts: the Gram path (exact) must reproduce
        // the dense thin-SVD path to float precision, embedding and ratios.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let (n, m) = (600, 40);
        let mut dense = vec![0.0_f32; n * m];
        for x in &mut dense {
            if rng.random::<f32>() < 0.2 {
                *x = (rng.random::<f32>() * 6.0).floor() + 1.0;
            }
        }
        let csr = CsrMatrix::from_dense_row_major(&dense, n, m);
        let opts = PcaOptions {
            n_components: 6,
            ..Default::default()
        };
        let exact = pca_dense(&csr.to_dense(), &opts).unwrap();
        let gram = pca_gram(&csr, &opts).unwrap();
        for j in 0..6 {
            let (a, b) = (gram.variance_ratio[j], exact.variance_ratio[j]);
            assert!((a - b).abs() < 1e-4, "vr {j}: {a} vs {b}");
        }
        // Columns agree up to sign.
        for j in 0..6 {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for i in 0..n {
                let (x, y) = (
                    f64::from(gram.embedding[i * 6 + j]),
                    f64::from(exact.embedding[i * 6 + j]),
                );
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let cos = dot.abs() / (na.sqrt() * nb.sqrt());
            assert!(cos > 0.9999, "component {j}: |cos| = {cos}");
        }
        let again = pca_gram(&csr, &opts).unwrap();
        assert_eq!(
            gram.embedding, again.embedding,
            "Gram PCA must be bit-identical"
        );
    }

    #[test]
    fn pca_dispatches_on_matrix_enum() {
        let d = rank_two_dense();
        let m = Matrix::Dense(d.clone());
        let result = pca(
            &m,
            &PcaOptions {
                n_components: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.n_components, 2);

        let csr = CsrMatrix::from_dense_row_major(&d.data, d.nrows, d.ncols);
        let m = Matrix::Csr(csr);
        let result = pca(
            &m,
            &PcaOptions {
                n_components: 2,
                oversample: 5,
                n_iter: 2,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.n_components, 2);
    }

    #[test]
    fn pca_is_deterministic_with_seed() {
        let d = rank_two_dense();
        let csr = CsrMatrix::from_dense_row_major(&d.data, d.nrows, d.ncols);
        let opts = PcaOptions {
            n_components: 2,
            oversample: 5,
            n_iter: 1,
            seed: 12345,
            center: false, // TruncatedSVD doesn't center
        };
        let r1 = pca(&Matrix::Csr(csr.clone()), &opts).unwrap();
        let r2 = pca(&Matrix::Csr(csr), &opts).unwrap();
        // Same seed → identical embedding.
        for (a, b) in r1.embedding.iter().zip(&r2.embedding) {
            assert!((a - b).abs() < 1e-5, "non-deterministic: {a} vs {b}");
        }
    }
}
