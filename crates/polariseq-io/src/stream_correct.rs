//! Streaming multi-dataset batch correction for large-scale integration.
//!
//! The architecture: never hold multiple raw matrices in RAM simultaneously.
//! Each dataset is processed independently via streaming (chunked reads from
//! .h5ad), producing only its small PCA embedding. The embeddings are then
//! merged and corrected (Harmony) — all in a fraction of the memory that
//! naive concatenation would require.
//!
//! ```text
//! Dataset A (1M cells) ──stream──> normalize ──> project ──> emb_A (1M × 50)
//! Dataset B (1M cells) ──stream──> normalize ──> project ──> emb_B (1M × 50)
//! Dataset C (1M cells) ──stream──> normalize ──> project ──> emb_C (1M × 50)
//!                                                                     │
//!              concat embeddings (3M × 50 = 600 MB) ←────────────────┘
//!                            │
//!              Harmony batch correction (subsampled k-means)
//!                            │
//!              neighbors → Leiden → UMAP (on corrected embedding)
//! ```
//!
//! Peak memory: `O(chunk_size` × `n_genes`) + `O(total_cells` × `n_pcs`)
//! For 4M cells total, 30K genes, 50K chunks: ~1 GB chunk + ~800 MB embeddings
//! = under 2 GB regardless of how many datasets.

#![allow(
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]

use crate::lazy::{read_rows_h5ad, scan_h5ad};
use crate::{IoError, Result};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Options for streaming batch correction.
#[derive(Debug, Clone)]
pub struct StreamingCorrectOptions {
    /// Number of PCA components to compute.
    pub n_components: usize,
    /// Oversampling for randomized SVD.
    pub oversample: usize,
    /// Chunk size for streaming reads (rows per chunk).
    pub chunk_size: usize,
    /// Target sum for `normalize_total`.
    pub target_sum: f32,
    /// Number of Harmony clusters.
    pub n_harmony_clusters: usize,
    /// Harmony max iterations.
    pub harmony_iterations: usize,
    /// Subsample size for Harmony k-means (0 = use all).
    pub harmony_subsample: usize,
    /// Number of neighbors for the kNN graph.
    pub n_neighbors: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for StreamingCorrectOptions {
    fn default() -> Self {
        Self {
            n_components: 50,
            oversample: 10,
            chunk_size: 50_000,
            target_sum: 1e4,
            n_harmony_clusters: 30,
            harmony_iterations: 10,
            harmony_subsample: 200_000,
            n_neighbors: 15,
            seed: 42,
        }
    }
}

/// Result of streaming batch correction.
#[derive(Debug, Clone)]
pub struct StreamingCorrectResult {
    /// Batch-corrected embedding (`total_cells` × `n_components`).
    pub embedding: Vec<f32>,
    /// Per-dataset batch labels (0-indexed).
    pub batch_labels: Vec<u32>,
    /// Per-dataset cell counts.
    pub dataset_sizes: Vec<usize>,
    /// Leiden cluster labels.
    pub cluster_labels: Vec<u32>,
    /// UMAP 2D coordinates (`total_cells` × 2).
    pub umap: Vec<f32>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Peak memory during streaming phase (bytes).
    pub peak_streaming_bytes: usize,
    /// Total elapsed time (seconds).
    pub elapsed_secs: f64,
}

/// Run the complete streaming batch correction pipeline on multiple .h5ad files.
///
/// Each dataset is processed independently (normalize → log1p → PCA via streaming),
/// then the embeddings are merged and corrected with Harmony. Never loads
/// multiple raw matrices simultaneously.
///
/// # Errors
/// Returns [`IoError`] on any read or computation failure.
#[allow(clippy::too_many_lines)]
pub fn streaming_merge_correct(
    paths: &[&str],
    opts: &StreamingCorrectOptions,
) -> Result<StreamingCorrectResult> {
    let t_start = std::time::Instant::now();

    // ── Phase 0: Scan all datasets for metadata (instant) ──────────
    let metas: Vec<_> = paths
        .iter()
        .map(|p| scan_h5ad(p))
        .collect::<Result<Vec<_>>>()?;
    let total_cells: usize = metas.iter().map(|m| m.n_cells).sum();

    // ── Gene intersection: handle datasets with different gene sets ──
    // Priority: (1) if all have names → intersection; (2) if same count → identity.
    let (n_genes, gene_maps): (usize, Vec<Vec<u32>>) =
        if metas.iter().all(|m| !m.gene_names.is_empty()) {
            // Gene names available — check if they match or need intersection.
            let first_names: std::collections::HashSet<&str> = metas[0]
                .gene_names
                .iter()
                .map(std::string::String::as_str)
                .collect();
            let all_match = metas.iter().all(|m| {
                m.gene_names.len() == metas[0].gene_names.len() && {
                    let names: std::collections::HashSet<&str> = m
                        .gene_names
                        .iter()
                        .map(std::string::String::as_str)
                        .collect();
                    names.len() == first_names.len() && names == first_names
                }
            });

            if all_match {
                // Same gene sets — identity mapping.
                let n = metas[0].n_genes;
                (
                    n,
                    metas
                        .iter()
                        .map(|_| (0..n as u32).collect::<Vec<u32>>())
                        .collect(),
                )
            } else {
                // Different gene sets — compute intersection.
                compute_gene_intersection(&metas)?
            }
        } else if metas.iter().all(|m| m.n_genes == metas[0].n_genes) {
            // Same count, no names available — assume same order (identity).
            let n = metas[0].n_genes;
            (n, metas.iter().map(|_| (0..n as u32).collect()).collect())
        } else {
            // Different counts and no names — cannot reconcile.
            return Err(IoError::InvalidLayout(format!(
                "gene count mismatch ({} vs {}) and no gene names for reconciliation",
                metas[0].n_genes,
                metas.iter().map(|m| m.n_genes).max().unwrap_or(0)
            )));
        };

    // ── Phase 1: Generate shared random projection Ω ───────────────
    // Using the SAME Ω for all datasets ensures the embeddings are in
    // the SAME random subspace — this is what makes them comparable
    // without joint PCA on the concatenated matrix.
    let l = (opts.n_components + opts.oversample).min(total_cells.min(n_genes));
    // Standard normal, not uniform: a uniform [0, 1) projection has a large
    // mean component, so the leading direction of the sketch is the mean
    // rather than structure. Shared across datasets by construction (one
    // seed), which is what puts every dataset in the same subspace.
    let omega = polariseq_core::rsvd::gaussian_rowmajor(n_genes, l, opts.seed);

    // ── Phase 2: Process each dataset independently (streaming) ────
    let mut all_embeddings = vec![0.0_f32; total_cells * l];
    let mut batch_labels = Vec::with_capacity(total_cells);
    let mut peak_streaming_bytes = 0_usize;
    let mut cell_offset = 0_usize;

    for (ds_idx, (path, meta)) in paths.iter().zip(&metas).enumerate() {
        let mut row_start = 0;
        while row_start < meta.n_cells {
            let row_end = (row_start + opts.chunk_size).min(meta.n_cells);
            let mut csr = read_rows_h5ad(path, row_start, row_end)?;
            peak_streaming_bytes = peak_streaming_bytes.max(csr.nnz() * 8 + csr.nrows * 4);

            // ── Normalize + log1p on this chunk (in-place) ──
            for i in 0..csr.nrows {
                let (s, e) = (csr.indptr[i] as usize, csr.indptr[i + 1] as usize);
                let row_sum: f32 = csr.data[s..e].iter().sum();
                if row_sum > 0.0 {
                    let scale = opts.target_sum / row_sum;
                    for v in &mut csr.data[s..e] {
                        *v = (*v * scale).ln_1p();
                    }
                }
            }

            // ── Project chunk · Ω → accumulate into embedding ──
            // Uses gene_maps[ds_idx] to remap original gene indices to the
            // shared intersection space (u32::MAX = gene not in intersection).
            let m = row_end - row_start;
            let base = (cell_offset + row_start) * l;
            let gene_map = &gene_maps[ds_idx];
            all_embeddings[base..base + m * l]
                .par_chunks_mut(l)
                .enumerate()
                .for_each(|(i, emb_row)| {
                    let (cols, vals) = csr.row(i);
                    for (&c, &v) in cols.iter().zip(vals) {
                        let mapped = gene_map[c as usize];
                        if mapped == u32::MAX {
                            continue; // gene not in intersection
                        }
                        let omega_row = &omega[(mapped as usize) * l..(mapped as usize + 1) * l];
                        for (er, &ov) in emb_row.iter_mut().zip(omega_row) {
                            *er += v * ov;
                        }
                    }
                });

            row_start = row_end;
        }

        // Assign batch labels for this dataset.
        batch_labels.extend(std::iter::repeat_n(ds_idx as u32, meta.n_cells));
        cell_offset += meta.n_cells;
    }

    // ── Phase 3: Orthonormalize the joint sketch → PCA ─────────────
    // The sketch Y (total_cells × l) captures the joint subspace.
    // QR gives Q (orthonormal basis), then we project back to get PCs.
    use faer::{linalg::solvers::Qr, Mat};
    let y_mat = Mat::<f32>::from_fn(total_cells, l, |i, j| all_embeddings[i * l + j]);
    let qr = Qr::new(y_mat.as_ref());
    let q = qr.compute_thin_Q(); // (total_cells × l)

    // Truncate to n_components.
    let k = opts.n_components.min(l);
    let mut pca_embedding = vec![0.0_f32; total_cells * k];
    for i in 0..total_cells {
        for j in 0..k {
            pca_embedding[i * k + j] = q[(i, j)];
        }
    }
    drop(y_mat);

    // ── Phase 4: Harmony batch correction on the embedding ─────────
    let corrected = subsampled_harmony(&pca_embedding, &batch_labels, total_cells, k, opts);

    // ── Phase 5: Neighbors + Leiden on corrected embedding ─────────
    let graph = polariseq_core::neighbors(
        &corrected,
        total_cells,
        k,
        &polariseq_core::NeighborsOptions {
            n_neighbors: opts.n_neighbors,
            ..Default::default()
        },
    )
    .map_err(|e| IoError::InvalidLayout(format!("neighbors failed: {e}")))?;

    let clustering = polariseq_core::louvain::fast_community(
        &graph,
        &polariseq_core::louvain::FastCommunityOptions {
            seed: opts.seed,
            ..Default::default()
        },
    );

    // ── Phase 6: UMAP ──────────────────────────────────────────────
    let umap_opts = polariseq_core::umap::UmapOptions {
        n_epochs: Some(200),
        seed: opts.seed,
        ..Default::default()
    };
    let umap_embedding = polariseq_core::umap::umap(&graph, &umap_opts);

    let dataset_sizes: Vec<usize> = metas.iter().map(|m| m.n_cells).collect();

    Ok(StreamingCorrectResult {
        embedding: corrected,
        batch_labels,
        dataset_sizes,
        cluster_labels: clustering.labels,
        umap: umap_embedding,
        n_clusters: clustering.n_communities,
        peak_streaming_bytes,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    })
}

/// Compute the intersection of genes across datasets and build per-dataset
/// remapping arrays (original index → intersection index, `u32::MAX` for absent).
fn compute_gene_intersection(metas: &[crate::lazy::DatasetMeta]) -> Result<(usize, Vec<Vec<u32>>)> {
    use std::collections::HashMap;

    if metas.is_empty() || metas.iter().any(|m| m.gene_names.is_empty()) {
        return Err(IoError::InvalidLayout(
            "gene names not available for intersection".into(),
        ));
    }

    // Count gene presence across datasets.
    let mut gene_count: HashMap<&str, usize> = HashMap::new();
    for m in metas {
        let seen: std::collections::HashSet<&str> = m
            .gene_names
            .iter()
            .map(std::string::String::as_str)
            .collect();
        for name in seen {
            *gene_count.entry(name).or_insert(0) += 1;
        }
    }

    // Keep genes present in ALL datasets.
    let n_datasets = metas.len();
    let common: Vec<&str> = gene_count
        .into_iter()
        .filter(|(_, count)| *count == n_datasets)
        .map(|(name, _)| name)
        .collect();

    if common.is_empty() {
        return Err(IoError::InvalidLayout(
            "no common genes across datasets".into(),
        ));
    }

    // Build the intersection index (sorted for determinism).
    let mut sorted_common: Vec<&str> = common;
    sorted_common.sort_unstable();
    let intersection_index: HashMap<&str, u32> = sorted_common
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i as u32))
        .collect();

    // Build per-dataset remapping.
    let maps: Vec<Vec<u32>> = metas
        .iter()
        .map(|m| {
            m.gene_names
                .iter()
                .map(|name| {
                    intersection_index
                        .get(name.as_str())
                        .copied()
                        .unwrap_or(u32::MAX)
                })
                .collect()
        })
        .collect();

    Ok((sorted_common.len(), maps))
}

/// Subsampled Harmony: k-means on a subsample, then apply correction to all.
///
/// Standard Harmony runs k-means on all n cells — O(n × k × d × iters).
/// For 4M cells × 30 clusters × 50 dims × 10 iters, that's 60 billion ops.
/// Subsampling to 200K for k-means reduces this by 20× with minimal quality
/// loss (the centroids are statistically equivalent).
fn subsampled_harmony(
    embedding: &[f32],
    batch_labels: &[u32],
    n: usize,
    d: usize,
    opts: &StreamingCorrectOptions,
) -> Vec<f32> {
    let k_harmony = opts.n_harmony_clusters;
    let subsample_size = if opts.harmony_subsample > 0 && opts.harmony_subsample < n {
        opts.harmony_subsample
    } else {
        n
    };

    // ── Subsample for k-means ──────────────────────────────────────
    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    let sample_indices: Vec<usize> = if subsample_size < n {
        use rand::seq::index::sample;
        sample(&mut rng, n, subsample_size).into_vec()
    } else {
        (0..n).collect()
    };

    // ── K-means on the subsample ───────────────────────────────────
    let sample_emb: Vec<f32> = sample_indices
        .iter()
        .flat_map(|&i| embedding[i * d..(i + 1) * d].to_vec())
        .collect();

    // Run k-means to find cluster centroids.
    let centroids = kmeans(&sample_emb, subsample_size, d, k_harmony, opts.seed);

    // ── Assign ALL cells to nearest centroid (parallel) ────────────
    let assignments: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let point = &embedding[i * d..(i + 1) * d];
            let mut best = 0_u32;
            let mut best_dist = f32::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let mut dist = 0.0_f32;
                for j in 0..d {
                    let diff = point[j] - centroid[j];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = c as u32;
                }
            }
            best
        })
        .collect();

    // ── Compute per-cluster-per-batch corrections ──────────────────
    let n_batches = (batch_labels.iter().max().copied().unwrap_or(0) as usize) + 1;
    let damping = 0.5_f32;

    let mut corrected = embedding.to_vec();

    for _iter in 0..opts.harmony_iterations {
        // Compute global cluster centroids and per-batch centroids.
        let mut global_centroids = vec![vec![0.0_f64; d]; k_harmony];
        let mut global_counts = vec![0_usize; k_harmony];
        let mut batch_centroids = vec![vec![vec![0.0_f64; d]; n_batches]; k_harmony];
        let mut batch_counts = vec![vec![0_usize; n_batches]; k_harmony];

        for i in 0..n {
            let c = assignments[i] as usize;
            let b = batch_labels[i] as usize;
            for j in 0..d {
                global_centroids[c][j] += f64::from(embedding[i * d + j]);
                batch_centroids[c][b][j] += f64::from(embedding[i * d + j]);
            }
            global_counts[c] += 1;
            batch_counts[c][b] += 1;
        }

        for c in 0..k_harmony {
            for j in 0..d {
                if global_counts[c] > 0 {
                    global_centroids[c][j] /= global_counts[c] as f64;
                }
            }
        }

        // Apply correction to each cell.
        corrected
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(i, cell)| {
                let c = assignments[i] as usize;
                let b = batch_labels[i] as usize;
                let count = batch_counts[c][b];
                if count > 0 {
                    for j in 0..d {
                        let batch_mean = batch_centroids[c][b][j] / count as f64;
                        let global_mean = global_centroids[c][j];
                        let correction = (global_mean - batch_mean) * f64::from(damping);
                        cell[j] += correction as f32;
                    }
                }
            });
    }

    corrected
}

/// Simple k-means with k-means++ initialization.
fn kmeans(data: &[f32], n: usize, d: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1);
    let k = k.min(n);

    // K-means++ init.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.random_range(0..n);
    centroids.push(data[first * d..(first + 1) * d].to_vec());

    for _ in 1..k {
        let dists: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data[i * d..(i + 1) * d];
                centroids
                    .iter()
                    .map(|c| {
                        let mut s = 0.0_f64;
                        for j in 0..d {
                            let diff = f64::from(point[j] - c[j]);
                            s += diff * diff;
                        }
                        s
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(f64::INFINITY)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total == 0.0 {
            let idx = rng.random_range(0..n);
            centroids.push(data[idx * d..(idx + 1) * d].to_vec());
        } else {
            let r = rng.random::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = n - 1;
            for (i, &dist) in dists.iter().enumerate() {
                cum += dist;
                if cum >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.push(data[chosen * d..(chosen + 1) * d].to_vec());
        }
    }

    // Lloyd's iterations.
    let mut assignments = vec![0_u32; n];
    for _ in 0..20 {
        // Assign.
        let new_assignments: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data[i * d..(i + 1) * d];
                let mut best = 0_u32;
                let mut best_dist = f64::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let mut s = 0.0_f64;
                    for j in 0..d {
                        let diff = f64::from(point[j] - centroid[j]);
                        s += diff * diff;
                    }
                    if s < best_dist {
                        best_dist = s;
                        best = c as u32;
                    }
                }
                best
            })
            .collect();

        let changed = new_assignments != assignments;
        assignments = new_assignments;

        // Update.
        let mut sums = vec![vec![0.0_f64; d]; k];
        let mut counts = vec![0_usize; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            for j in 0..d {
                sums[c][j] += f64::from(data[i * d + j]);
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c][j] = (sums[c][j] / counts[c] as f64) as f32;
                }
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_finds_separated_clusters() {
        // 3 well-separated clusters in 2D.
        let n = 300;
        let d = 2;
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let cluster = i % 3;
            data[i * 2] = cluster as f32 * 10.0 + 0.1;
            data[i * 2 + 1] = cluster as f32 * 10.0 + 0.1;
        }
        let centroids = kmeans(&data, n, d, 3, 42);
        assert_eq!(centroids.len(), 3);
        // Centroids should be near (0,0), (10,10), (20,20).
        let mut found_origins = 0;
        for c in &centroids {
            let dist_to_origin = (c[0] - 0.0).abs() + (c[1] - 0.0).abs();
            let dist_to_10 = (c[0] - 10.0).abs() + (c[1] - 10.0).abs();
            let dist_to_20 = (c[0] - 20.0).abs() + (c[1] - 20.0).abs();
            if dist_to_origin < 2.0 || dist_to_10 < 2.0 || dist_to_20 < 2.0 {
                found_origins += 1;
            }
        }
        assert!(found_origins >= 2, "kmeans didn't find cluster centers");
    }

    #[test]
    fn streaming_correct_two_small_datasets() {
        // Use the same dataset twice (simulating two batches of the same data).
        let paths = [
            "../../tests/data/sparse_100x200.h5ad",
            "../../tests/data/sparse_100x200.h5ad",
        ];
        let opts = StreamingCorrectOptions {
            n_components: 5,
            oversample: 5,
            chunk_size: 50,
            n_harmony_clusters: 3,
            harmony_iterations: 3,
            harmony_subsample: 0,
            n_neighbors: 5,
            ..Default::default()
        };
        let result = streaming_merge_correct(&paths, &opts).unwrap();
        let total: usize = result.dataset_sizes.iter().sum();
        assert_eq!(total, 200); // 100 + 100
        assert_eq!(result.embedding.len(), total * 5);
        assert_eq!(result.batch_labels.len(), total);
        assert_eq!(result.dataset_sizes.len(), 2);
        assert_eq!(result.dataset_sizes, vec![100, 100]);
        assert!(result.n_clusters >= 1);
    }
}
