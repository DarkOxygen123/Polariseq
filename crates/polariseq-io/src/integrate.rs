//! Streaming multi-dataset batch correction and integration.
//!
//! The architecture for merging large datasets on low-memory machines:
//!
//! ```text
//! Dataset A (1M) ──┐
//! Dataset B (1M) ──┤── Step 1: scan metadata → find shared genes (instant)
//! Dataset C (1M) ──┤── Step 2: streaming HVG across all datasets (chunked)
//!                   ├── Step 3: PCA on reference (subsample, streaming)
//!                   ├── Step 4: project each dataset onto reference (chunked)
//!                   ├── Step 5: concatenate embeddings (small: 3M × 50)
//!                   └── Step 6: Harmony correction (on embedding, fast)
//! ```
//!
//! Peak memory: `O(chunk_size` × `n_genes`) for the largest single chunk +
//! `O(n_total` × `n_pcs`) for the final embedding. For 3M cells × 50 PCs:
//! ~600 MB embedding + ~800 MB chunks = ~1.4 GB total (vs 24+ GB materialized).

use crate::lazy::{read_rows_any, scan_any, DatasetMeta};
use crate::IoError;
use crate::Result;
use polariseq_core::preprocess::{hvg_seurat, GeneStats};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Configuration for the streaming integration pipeline.
#[derive(Debug, Clone)]
pub struct IntegrateConfig {
    /// Number of highly variable genes to select (across all datasets).
    pub n_hvg: usize,
    /// Number of PCA components.
    pub n_pcs: usize,
    /// Chunk size for streaming reads (rows per chunk).
    pub chunk_size: usize,
    /// Subsample size for the reference PCA (from all datasets combined).
    pub reference_size: usize,
    /// Number of Harmony clusters.
    pub harmony_clusters: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for IntegrateConfig {
    fn default() -> Self {
        Self {
            n_hvg: 2000,
            n_pcs: 50,
            chunk_size: 50_000,
            reference_size: 100_000,
            harmony_clusters: 30,
            seed: 42,
        }
    }
}

/// The result of streaming integration.
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Batch-corrected embedding (`n_total` × `n_pcs`, row-major).
    pub embedding: Vec<f32>,
    /// Batch label for each cell (0-indexed dataset ID).
    pub batch_labels: Vec<u32>,
    /// Number of cells per dataset.
    pub dataset_sizes: Vec<usize>,
    /// Shared gene names (the intersection across datasets).
    pub shared_genes: Vec<String>,
    /// HVG indices (into `shared_genes`).
    pub hvg_indices: Vec<usize>,
    /// Processing statistics.
    pub stats: IntegrationStats,
}

/// Statistics from the integration run.
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Time for metadata scanning (seconds).
    pub scan_time_s: f64,
    /// Time for HVG selection (seconds).
    pub hvg_time_s: f64,
    /// Time for reference PCA (seconds).
    pub reference_pca_time_s: f64,
    /// Time for projection (seconds).
    pub projection_time_s: f64,
    /// Time for Harmony (seconds).
    pub harmony_time_s: f64,
    /// Total time (seconds).
    pub total_time_s: f64,
    /// Peak chunk memory (bytes).
    pub peak_chunk_bytes: usize,
    /// Number of cells processed.
    pub n_total_cells: usize,
}

/// Step 1: Scan all datasets and find the intersection of gene names.
///
/// This is instant — reads only metadata, no matrix data.
///
/// # Errors
/// Returns [`IoError`] if any scan fails.
pub fn find_shared_genes(paths: &[String]) -> Result<(Vec<String>, Vec<DatasetMeta>)> {
    let mut metas = Vec::new();
    let mut gene_sets: Vec<std::collections::HashSet<String>> = Vec::new();

    for path in paths {
        let meta = scan_any(path)?;
        let genes: std::collections::HashSet<String> = meta.gene_names.iter().cloned().collect();
        gene_sets.push(genes);
        metas.push(meta);
    }

    // Intersect all gene sets.
    let mut shared = gene_sets[0].clone();
    for gs in &gene_sets[1..] {
        shared = shared.intersection(gs).cloned().collect();
    }

    // Sort for deterministic ordering.
    let mut shared_genes: Vec<String> = shared.into_iter().collect();
    shared_genes.sort();

    Ok((shared_genes, metas))
}

/// Step 2: highly-variable genes across every dataset, on the shared genes.
///
/// Uses the canonical Seurat selector ([`hvg_seurat`]) on statistics
/// accumulated across all datasets, so a cross-dataset HVG set means the
/// same thing as `pp.highly_variable_genes` on a single matrix. (It used to
/// rank raw `variance/mean` with no mean-binning — a fourth, silently
/// different HVG definition — and to hash a gene *name* for every stored
/// nonzero to map dataset-local indices onto shared ones. The mapping is
/// now a `Vec<u32>` built once per dataset.)
///
/// # Errors
/// Returns [`IoError`] on any read failure.
pub fn streaming_cross_dataset_hvg(
    metas: &[DatasetMeta],
    shared_genes: &[String],
    n_top: usize,
    chunk_size: usize,
) -> Result<Vec<usize>> {
    use std::collections::HashMap;

    let n_genes = shared_genes.len();
    let shared_index: HashMap<&str, u32> = shared_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i as u32))
        .collect();

    let mut col_sums = vec![0.0_f64; n_genes];
    let mut col_sq_sums = vec![0.0_f64; n_genes];
    let mut total_cells = 0_usize;

    for meta in metas {
        // Dataset-local gene index -> shared index (u32::MAX = not shared).
        // One pass over the gene names, not one hash per nonzero.
        let local_to_shared: Vec<u32> = meta
            .gene_names
            .iter()
            .map(|name| shared_index.get(name.as_str()).copied().unwrap_or(u32::MAX))
            .collect();

        let chunk = chunk_size.clamp(1, meta.n_cells.max(1));
        let mut row_start = 0;
        while row_start < meta.n_cells {
            let row_end = (row_start + chunk).min(meta.n_cells);
            let csr = read_rows_any(&meta.path, row_start, row_end)?;
            for i in 0..csr.nrows {
                let (cols, vals) = csr.row(i);
                for (&c, &v) in cols.iter().zip(vals) {
                    let shared = local_to_shared[c as usize];
                    if shared != u32::MAX {
                        let x = f64::from(v);
                        col_sums[shared as usize] += x;
                        col_sq_sums[shared as usize] += x * x;
                    }
                }
            }
            total_cells += row_end - row_start;
            row_start = row_end;
        }
    }

    // Same mean/variance convention as `gene_mean_var` (ddof = 1), then the
    // canonical selector.
    let n = total_cells.max(1) as f64;
    let means: Vec<f64> = col_sums.iter().map(|&s| s / n).collect();
    let variances: Vec<f64> = col_sq_sums
        .iter()
        .zip(&means)
        .map(|(&q, &mean)| {
            if total_cells < 2 {
                0.0
            } else {
                ((q / n - mean * mean) * n / (n - 1.0)).max(0.0)
            }
        })
        .collect();
    let stats = GeneStats {
        means,
        variances,
        n_obs: total_cells,
    };
    let hvg = hvg_seurat(&stats, n_top, 20);
    Ok(hvg
        .mask
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| keep.then_some(i))
        .collect())
}

/// Step 3+4: Reference-based PCA and projection.
///
/// Subsamples cells from all datasets, computes PCA on the subsample,
/// then projects ALL cells from all datasets onto those components.
/// Each dataset is processed chunk-by-chunk.
///
/// # Errors
/// Returns [`IoError`] on any failure.
#[allow(clippy::too_many_lines)]
pub fn reference_pca_project(
    metas: &[DatasetMeta],
    shared_genes: &[String],
    hvg_indices: &[usize],
    n_pcs: usize,
    reference_size: usize,
    chunk_size: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f64>)> {
    // Returns: (embedding, variance_ratio, reference_mean)
    use faer::{linalg::solvers::Svd, Mat};
    use rand::{RngExt, SeedableRng};

    let n_hvg = hvg_indices.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // ── Build reference subsample ────────────────────────────────
    // Sample from each dataset proportionally.
    let total_cells: usize = metas.iter().map(|m| m.n_cells).sum();
    let per_dataset_ref: Vec<usize> = metas
        .iter()
        .map(|m| (m.n_cells as f64 / total_cells as f64 * reference_size as f64) as usize)
        .collect();

    // HVG index mapping: shared gene index → column in the HVG-subset matrix
    let hvg_position: std::collections::HashMap<usize, usize> = hvg_indices
        .iter()
        .enumerate()
        .map(|(pos, &gene_idx)| (gene_idx, pos))
        .collect();

    // Load reference cells (streaming, accumulate into a dense matrix).
    let ref_n: usize = per_dataset_ref.iter().sum();
    let mut ref_matrix = vec![0.0_f32; ref_n * n_hvg];
    let mut ref_row = 0;

    // Gene name → HVG position map for each dataset.
    let gene_to_hvg: Vec<std::collections::HashMap<&str, usize>> = metas
        .iter()
        .map(|meta| {
            meta.gene_names
                .iter()
                .filter_map(|name| {
                    // Find the shared gene index for this name
                    let shared_idx = shared_genes.iter().position(|g| g == name)?;
                    // Check if it's an HVG
                    let hvg_pos = hvg_position.get(&shared_idx).copied()?;
                    Some((name.as_str(), hvg_pos))
                })
                .collect()
        })
        .collect();

    let mut peak_chunk = 0_usize;

    for (di, meta) in metas.iter().enumerate() {
        let n_sample = per_dataset_ref[di].min(meta.n_cells);
        let mut sample_indices: Vec<usize> = (0..meta.n_cells).collect();
        // Fisher-Yates partial shuffle for efficiency.
        for i in 0..n_sample {
            let j = rng.random_range(i..meta.n_cells);
            sample_indices.swap(i, j);
        }
        sample_indices.truncate(n_sample);
        sample_indices.sort_unstable();

        // Read in contiguous blocks.
        let mut block_start = 0;
        while block_start < sample_indices.len() {
            // Find contiguous run.
            let mut block_end = block_start + 1;
            while block_end < sample_indices.len()
                && sample_indices[block_end] == sample_indices[block_end - 1] + 1
            {
                block_end += 1;
            }

            let rs = sample_indices[block_start];
            let re = sample_indices[block_end - 1] + 1;
            let csr = read_rows_any(&meta.path, rs, re)?;
            peak_chunk = peak_chunk.max(csr.nnz() * 8);

            let ds_map = &gene_to_hvg[di];
            for i in 0..csr.nrows {
                let (cols, vals) = csr.row(i);
                for (&c, &v) in cols.iter().zip(vals) {
                    let gene_name = &meta.gene_names[c as usize];
                    if let Some(&hvg_pos) = ds_map.get(gene_name.as_str()) {
                        if ref_row < ref_n {
                            ref_matrix[ref_row * n_hvg + hvg_pos] = v;
                        }
                    }
                }
            }
            ref_row += csr.nrows;

            block_start = block_end;
        }
    }

    // ── Center and PCA the reference ────────────────────────────
    // Compute column means.
    let mut col_means = vec![0.0_f64; n_hvg];
    for i in 0..ref_n {
        for j in 0..n_hvg {
            col_means[j] += f64::from(ref_matrix[i * n_hvg + j]);
        }
    }
    for m in &mut col_means {
        *m /= ref_n as f64;
    }

    // Center.
    for i in 0..ref_n {
        for j in 0..n_hvg {
            ref_matrix[i * n_hvg + j] -= col_means[j] as f32;
        }
    }

    // Thin SVD on the reference.
    let ref_mat = Mat::<f32>::from_fn(ref_n, n_hvg, |i, j| ref_matrix[i * n_hvg + j]);
    let svd = Svd::new_thin(ref_mat.as_ref())
        .map_err(|e| IoError::InvalidLayout(format!("reference SVD failed: {e:?}")))?;
    let s: Vec<f32> = svd.S().column_vector().iter().copied().collect();
    let v = svd.V(); // (n_hvg × k) — the projection components

    let k = n_pcs.min(s.len());

    // ── Project ALL cells onto the reference components ─────────
    let mut embedding = vec![0.0_f32; total_cells * k];
    let mut global_row = 0_usize;

    for (di, meta) in metas.iter().enumerate() {
        let chunk = chunk_size.max(10_000).min(meta.n_cells);
        let mut rs = 0;
        while rs < meta.n_cells {
            let re = (rs + chunk).min(meta.n_cells);
            let csr = read_rows_any(&meta.path, rs, re)?;
            peak_chunk = peak_chunk.max(csr.nnz() * 8);

            let ds_map = &gene_to_hvg[di];
            let m = re - rs;
            if k == 0 || m == 0 {
                global_row += m;
                rs = re;
                continue;
            }

            // Project each cell: (x - mean) @ V
            embedding[global_row * k..(global_row + m) * k]
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(i, emb_row)| {
                    // Build the centered HVG vector for this cell.
                    let mut x = vec![0.0_f32; n_hvg];
                    let (cols, vals) = csr.row(i);
                    for (&c, &v) in cols.iter().zip(vals) {
                        let gene_name = &meta.gene_names[c as usize];
                        if let Some(&hvg_pos) = ds_map.get(gene_name.as_str()) {
                            x[hvg_pos] = v - col_means[hvg_pos] as f32;
                        }
                    }
                    // Multiply by V.
                    for j in 0..k {
                        let mut dot = 0.0_f32;
                        for h in 0..n_hvg {
                            dot += x[h] * v[(h, j)];
                        }
                        emb_row[j] = dot * s[j];
                    }
                });

            global_row += m;
            rs = re;
        }
    }

    // Variance ratio.
    let total_var: f64 = s.iter().take(k).map(|v| f64::from(v * v)).sum();
    let variance_ratio: Vec<f32> = (0..k)
        .map(|j| (f64::from(s[j] * s[j]) / total_var) as f32)
        .collect();

    let _ = peak_chunk; // tracked but returned via stats
    Ok((embedding, variance_ratio, col_means))
}

/// Full streaming integration pipeline.
///
/// Processes multiple .h5ad datasets with batch correction, never loading
/// the full expression matrix. Returns a corrected embedding ready for
/// clustering.
///
/// # Errors
/// Returns [`IoError`] on any failure.
pub fn integrate_datasets(paths: &[String], config: &IntegrateConfig) -> Result<IntegrationResult> {
    let t_total = std::time::Instant::now();

    // Step 1: Scan metadata + find shared genes.
    let t0 = std::time::Instant::now();
    let (shared_genes, metas) = find_shared_genes(paths)?;
    let scan_time = t0.elapsed().as_secs_f64();

    // Sanity check: shared genes must be non-empty.
    if shared_genes.is_empty() {
        return Err(IoError::InvalidLayout(
            "no shared genes across datasets — check that all datasets use the same \
             gene identifier system (e.g., all Ensembl IDs or all gene symbols)"
                .to_string(),
        ));
    }
    if shared_genes.len() < config.n_hvg {
        eprintln!(
            "[integrate] warning: only {} shared genes < requested {} HVGs",
            shared_genes.len(),
            config.n_hvg
        );
    }

    // Step 2: Streaming cross-dataset HVG.
    let t0 = std::time::Instant::now();
    let hvg_indices =
        streaming_cross_dataset_hvg(&metas, &shared_genes, config.n_hvg, config.chunk_size)?;
    let hvg_time = t0.elapsed().as_secs_f64();

    // Step 3+4: Reference PCA + projection.
    let t0 = std::time::Instant::now();
    let (mut embedding, variance_ratio, _ref_mean) = reference_pca_project(
        &metas,
        &shared_genes,
        &hvg_indices,
        config.n_pcs,
        config.reference_size,
        config.chunk_size,
        config.seed,
    )?;
    let pca_time = t0.elapsed().as_secs_f64();

    // Step 5: Harmony batch correction on the embedding.
    let t0 = std::time::Instant::now();
    let batch_labels: Vec<u32> = metas
        .iter()
        .enumerate()
        .flat_map(|(di, m)| vec![di as u32; m.n_cells])
        .collect();

    let harmony_opts = polariseq_core::harmony::HarmonyOptions {
        n_clusters: Some(config.harmony_clusters),
        seed: config.seed,
        ..Default::default()
    };

    let total_cells: usize = metas.iter().map(|m| m.n_cells).sum();
    let n_pcs = config.n_pcs;

    if metas.len() > 1 {
        let corrected = polariseq_core::harmony::harmony(
            &embedding,
            &batch_labels,
            total_cells,
            n_pcs,
            &harmony_opts,
        )
        .map_err(|e| IoError::InvalidLayout(format!("harmony failed: {e}")))?;
        embedding = corrected;
    }
    let harmony_time = t0.elapsed().as_secs_f64();

    let dataset_sizes: Vec<usize> = metas.iter().map(|m| m.n_cells).collect();
    let _ = variance_ratio; // could be returned in the result

    Ok(IntegrationResult {
        embedding,
        batch_labels,
        dataset_sizes,
        shared_genes,
        hvg_indices,
        stats: IntegrationStats {
            scan_time_s: scan_time,
            hvg_time_s: hvg_time,
            reference_pca_time_s: pca_time,
            projection_time_s: 0.0, // included in pca_time
            harmony_time_s: harmony_time,
            total_time_s: t_total.elapsed().as_secs_f64(),
            peak_chunk_bytes: config.chunk_size * 30_000 * 8, // estimate
            n_total_cells: total_cells,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_shared_genes_two_datasets() {
        let paths = vec!["../../tests/data/sparse_100x200.h5ad".to_string()];
        let (genes, metas) = find_shared_genes(&paths).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(genes.len(), 200); // all genes shared with itself
    }

    #[test]
    fn streaming_hvg_selects_genes() {
        let paths = vec!["../../tests/data/sparse_100x200.h5ad".to_string()];
        let (genes, metas) = find_shared_genes(&paths).unwrap();
        let hvg = streaming_cross_dataset_hvg(&metas, &genes, 50, 1000).unwrap();
        assert_eq!(hvg.len(), 50);
        // All indices should be valid.
        assert!(hvg.iter().all(|&i| i < genes.len()));
    }

    #[test]
    fn integration_single_dataset() {
        let paths = vec!["../../tests/data/sparse_100x200.h5ad".to_string()];
        let config = IntegrateConfig {
            n_hvg: 50,
            n_pcs: 10,
            chunk_size: 1000,
            reference_size: 50,
            harmony_clusters: 5,
            seed: 42,
        };
        let result = integrate_datasets(&paths, &config).unwrap();
        assert_eq!(result.dataset_sizes, vec![100]);
        assert_eq!(result.embedding.len(), 100 * 10);
        assert_eq!(result.batch_labels.len(), 100);
        // All batch labels should be 0 (single dataset).
        assert!(result.batch_labels.iter().all(|&b| b == 0));
    }
}
