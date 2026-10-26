//! Per-cluster statistics over a file, without materialising it.
//!
//! This module used to carry its own chunk-reading loop and its own
//! accumulators. It is now a thin adapter: [`H5adStream`] turns the file
//! into row blocks and [`cluster_stats_blocks`] does the arithmetic — the
//! same kernel the in-memory path uses, so annotation and marker ranking
//! give the same answers whether the matrix is in RAM or on disk.

use polariseq_core::preprocess::{
    cluster_stats_blocks, row_totals_blocks, ClusterStats, NormalizedView,
};
use polariseq_core::rowblocks::RowBlocks;

use crate::stream::H5adStream;
use crate::Result;

/// Per-cluster, per-gene statistics for the matrix in `path`.
///
/// Memory is `O(n_genes × n_clusters)` for the accumulators plus one block
/// of rows.
///
/// # Errors
/// Returns [`crate::IoError`] if the file cannot be scanned or read.
pub fn streaming_cluster_stats(
    path: &str,
    labels: &[u32],
    n_clusters: usize,
    chunk_size: usize,
) -> Result<ClusterStats> {
    let stream = H5adStream::open(path)?;
    if labels.len() != stream.n_rows() {
        return Err(crate::IoError::InvalidLayout(format!(
            "labels length {} != n_cells {}",
            labels.len(),
            stream.n_rows()
        )));
    }
    // `RowBlocks::for_each_block` cannot report an error, so read one row
    // up front: an unreadable or malformed file fails here rather than
    // silently yielding zeroed statistics. `chunk_size` is accepted for
    // API compatibility; the kernel picks its own block size.
    let _ = chunk_size;
    if stream.n_rows() > 0 {
        crate::lazy::read_rows_any(path, 0, 1)?;
    }
    Ok(cluster_stats_blocks(&stream, labels, n_clusters))
}

/// Mean expression per (gene, cluster): `(means, n_genes, n_clusters)` with
/// `means` laid out `[gene * n_clusters + cluster]`.
///
/// # Errors
/// Returns [`crate::IoError`] on read failure.
pub fn streaming_cluster_means(
    path: &str,
    labels: &[u32],
    n_clusters: usize,
    chunk_size: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let stats = streaming_cluster_stats(path, labels, n_clusters, chunk_size)?;
    let means = stats.means();
    Ok((means, stats.n_genes, n_clusters))
}

/// Per-cluster, per-gene statistics over the normalize-total + log1p
/// transform of the matrix in `path`, without materialising it.
///
/// Marker ranking and annotation are statements about the *normalised*
/// matrix — the same transform the pipeline applies before PCA — so offering
/// only raw-count statistics here would make an out-of-core run and an
/// in-memory run of the same file answer different questions. The values
/// pass through the same [`NormalizedView`] the in-memory pipeline uses, so
/// the statistics equal an in-memory run's exactly.
///
/// Two passes over the file: the first collects each row's total (the
/// divisor must exist before any value can be scaled), the second
/// accumulates. Memory is `O(n_rows + n_genes × n_clusters)` plus one block.
/// As in [`cluster_stats_blocks`], label ids `>= n_clusters` are ignored, so
/// a caller holding labels only for quality-filtered cells passes a sentinel
/// for the rest.
///
/// # Errors
/// Returns [`crate::IoError`] if the file cannot be scanned or read.
pub fn streaming_cluster_stats_normalized(
    path: &str,
    labels: &[u32],
    n_clusters: usize,
    target_sum: f32,
) -> Result<ClusterStats> {
    streaming_cluster_stats_normalized_masked(path, labels, n_clusters, target_sum, None)
}

/// [`streaming_cluster_stats_normalized`] for a gene-filtered analysis: each
/// row is normalised by its total over the genes flagged in `gene_keep`, as
/// `normalize_total` would normalise the matrix with the other genes
/// removed. Statistics are still returned for every gene; the caller drops
/// the removed ones (see `de::rank_genes_groups_from_stats`).
///
/// # Errors
/// Returns [`crate::IoError`] if the file cannot be scanned or read, or if
/// `labels` or `gene_keep` do not match the file's shape.
pub fn streaming_cluster_stats_normalized_masked(
    path: &str,
    labels: &[u32],
    n_clusters: usize,
    target_sum: f32,
    gene_keep: Option<&[bool]>,
) -> Result<ClusterStats> {
    let stream = H5adStream::open(path)?;
    if labels.len() != stream.n_rows() {
        return Err(crate::IoError::InvalidLayout(format!(
            "labels length {} != n_cells {}",
            labels.len(),
            stream.n_rows()
        )));
    }
    if let Some(keep) = gene_keep {
        if keep.len() != stream.n_cols() {
            return Err(crate::IoError::InvalidLayout(format!(
                "gene mask length {} != n_genes {}",
                keep.len(),
                stream.n_cols()
            )));
        }
    }
    if stream.n_rows() > 0 {
        crate::lazy::read_rows_any(path, 0, 1)?;
    }
    // Pass 1: row totals (summed as `normalize_total` sums them); pass 2:
    // accumulate the normalised view.
    let totals = row_totals_blocks(&stream, gene_keep);
    let scale = NormalizedView::scale_from_totals(&totals, target_sum);
    let view = NormalizedView::new(&stream, scale, true);
    Ok(cluster_stats_blocks(&view, labels, n_clusters))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polariseq_core::CsrMatrix;

    fn fixture() -> String {
        format!(
            "{}/../../tests/data/sparse_100x200.h5ad",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn in_ram() -> CsrMatrix {
        let f = crate::AnnDataFile::open(fixture()).unwrap();
        match f.read_x().unwrap() {
            polariseq_core::Matrix::Csr(c) => c,
            polariseq_core::Matrix::Dense(_) => unreachable!(),
        }
    }

    /// Streaming the file must give exactly what the in-memory kernel gives.
    #[test]
    fn streamed_cluster_stats_match_in_ram() {
        let labels: Vec<u32> = (0..100).map(|i| (i % 3) as u32).collect();
        let streamed = streaming_cluster_stats(&fixture(), &labels, 3, 32).unwrap();
        let direct = cluster_stats_blocks(&in_ram(), &labels, 3);
        assert_eq!(streamed, direct);
    }

    /// Means computed the long way (dense, per cluster) must agree.
    #[test]
    fn cluster_means_match_a_dense_reference() {
        let labels: Vec<u32> = (0..100).map(|i| (i % 4) as u32).collect();
        let (means, n_genes, n_clusters) =
            streaming_cluster_means(&fixture(), &labels, 4, 64).unwrap();
        assert_eq!((n_genes, n_clusters), (200, 4));

        let dense = in_ram().to_dense();
        for c in 0..4 {
            let rows: Vec<usize> = (0..100).filter(|i| labels[*i] as usize == c).collect();
            for g in [0_usize, 7, 199] {
                let want: f64 = rows
                    .iter()
                    .map(|&i| f64::from(dense.row(i)[g]))
                    .sum::<f64>()
                    / rows.len() as f64;
                let got = f64::from(means[g * 4 + c]);
                assert!(
                    (got - want).abs() < 1e-5 * want.abs().max(1.0),
                    "gene {g} cluster {c}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn rejects_mismatched_labels() {
        assert!(streaming_cluster_stats(&fixture(), &[0, 1, 2], 3, 32).is_err());
    }

    /// Normalised streaming statistics must equal the in-memory kernel run
    /// on the same normalised view — the parity that makes an out-of-core
    /// marker ranking comparable to an in-memory one.
    #[test]
    fn normalized_streamed_stats_match_in_ram() {
        let labels: Vec<u32> = (0..100).map(|i| (i % 3) as u32).collect();
        let streamed =
            streaming_cluster_stats_normalized(&fixture(), &labels, 3, 10_000.0).unwrap();

        let csr = in_ram();
        let totals = csr.row_sums();
        let scale = NormalizedView::scale_from_totals(&totals, 10_000.0);
        let view = NormalizedView::new(&csr, scale, true);
        let direct = cluster_stats_blocks(&view, &labels, 3);
        assert_eq!(streamed, direct);
    }

    /// Sentinel labels (`>= n_clusters`) are excluded — the caller's way of
    /// holding labels only for quality-filtered cells.
    #[test]
    fn normalized_stats_skip_sentinel_rows() {
        let labels: Vec<u32> = (0..100)
            .map(|i| {
                if i % 10 == 0 {
                    u32::MAX
                } else {
                    (i % 3) as u32
                }
            })
            .collect();
        let stats = streaming_cluster_stats_normalized(&fixture(), &labels, 3, 10_000.0).unwrap();
        assert_eq!(stats.cluster_sizes, vec![30, 30, 30]);
    }
}
