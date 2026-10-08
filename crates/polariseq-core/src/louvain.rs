//! Fast community detection: multi-level Louvain on CSR.
//!
//! The engine lives in [`crate::leiden`]: queue-based local moving and
//! counting-sort aggregation, without Leiden's refinement pass. This runs
//! on the same weighted connectivity graph as [`crate::clustering::leiden`]
//! (the UMAP fuzzy connectivities — the graph igraph receives too), so the
//! reported modularity is directly comparable across methods. Kept as the
//! explicit `method="fast"` option for very large graphs.

use crate::backend::NeighborGraph;
use crate::leiden;

/// Options for fast community detection.
#[derive(Debug, Clone)]
pub struct FastCommunityOptions {
    /// Resolution parameter γ. Higher → more, smaller communities.
    pub resolution: f64,
    /// Cap on aggregation levels (the old per-pass cap; local moving now
    /// converges on its own via the node queue).
    pub max_iterations: usize,
    /// Stop after a level that moved fewer than this fraction of nodes.
    pub convergence_threshold: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for FastCommunityOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 10,
            convergence_threshold: 0.001,
            seed: 0,
        }
    }
}

/// Result of community detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CommunityResult {
    /// Community label for each node (contiguous, size-descending).
    pub labels: Vec<u32>,
    /// Number of communities.
    pub n_communities: usize,
    /// Modularity of the partition on the weighted connectivity graph —
    /// the same graph igraph and [`crate::clustering::leiden`] evaluate.
    pub modularity: f64,
    /// Undirected edge count of that graph.
    pub n_edges: usize,
    /// Total edge weight of that graph (m).
    pub total_weight: f64,
}

/// Run multi-level Louvain on the connectivity graph of a kNN distance
/// graph. Deterministic for a given seed.
///
/// # Panics
/// Panics if the graph is empty.
#[must_use]
pub fn fast_community(graph: &NeighborGraph, opts: &FastCommunityOptions) -> CommunityResult {
    assert!(graph.n_obs > 0, "empty graph");
    communities(&leiden::weighted_from_distances(graph), opts)
}

/// [`fast_community`] on a symmetric connectivity CSR used as given (see
/// [`crate::clustering::leiden_connectivities`]).
///
/// # Panics
/// Panics if the graph is empty.
#[must_use]
pub fn fast_community_connectivities(
    graph: NeighborGraph,
    opts: &FastCommunityOptions,
) -> CommunityResult {
    assert!(graph.n_obs > 0, "empty graph");
    communities(&leiden::weighted_from_connectivities(graph), opts)
}

/// [`fast_community_connectivities`] on a connectivity CSR the caller holds,
/// read in place (nothing is copied).
///
/// # Panics
/// Panics if the graph is empty.
#[must_use]
pub fn fast_community_connectivities_ref(
    indptr: &[u32],
    indices: &[u32],
    data: &[f32],
    n_obs: usize,
    opts: &FastCommunityOptions,
) -> CommunityResult {
    assert!(n_obs > 0, "empty graph");
    communities(
        &leiden::WeightedGraph::from_symmetric_csr_ref(indptr, indices, data, n_obs),
        opts,
    )
}

fn communities(wg: &leiden::WeightedGraph<'_>, opts: &FastCommunityOptions) -> CommunityResult {
    let labels = leiden::louvain_partition(
        wg,
        opts.resolution,
        opts.seed,
        opts.max_iterations.clamp(1, 10),
        opts.convergence_threshold,
    );
    let modularity = leiden::modularity(wg, &labels, opts.resolution);
    let (labels, n_communities) = leiden::relabel_by_size(&labels);
    CommunityResult {
        labels,
        n_communities,
        modularity,
        n_edges: wg.n_undirected,
        total_weight: wg.two_m / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(n: usize, edges: &[(usize, usize, f32)]) -> NeighborGraph {
        use std::collections::BTreeMap;
        let mut adj: BTreeMap<usize, Vec<(u32, f32)>> = BTreeMap::new();
        for &(s, d, w) in edges {
            adj.entry(s).or_default().push((d as u32, w));
            adj.entry(d).or_default().push((s as u32, w));
        }
        let mut indptr = vec![0_u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            if let Some(nbrs) = adj.get(&i) {
                for &(idx, w) in nbrs {
                    indices.push(idx);
                    data.push(w);
                }
            }
            indptr.push(indices.len() as u32);
        }
        NeighborGraph {
            indptr,
            indices,
            data,
            n_obs: n,
        }
    }

    #[test]
    fn finds_two_clusters() {
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (1, 2, 0.1),
            (3, 4, 0.1),
            (3, 5, 0.1),
            (4, 5, 0.1),
            (2, 3, 5.0), // weak bridge (large distance → weak fuzzy weight)
        ];
        let g = make_graph(6, &edges);
        let result = fast_community(&g, &FastCommunityOptions::default());
        assert_eq!(result.n_communities, 2, "should find 2 clusters");
        let c0 = result.labels[0];
        assert_eq!(result.labels[1], c0);
        assert_eq!(result.labels[2], c0);
        assert_ne!(result.labels[3], c0);
        // Exact for this fixture: node 2's two nearest neighbours sit at ρ and
        // already reach umap-learn's target log2(4) = 2, so σ falls to its
        // floor and the long bridge's weight to ~0 (pruned): two triangles,
        // m = 6, Q = 1 − 2·(1/2)² = 0.5. (Before the 2026-09-20 fix, σ
        // diverged instead and the bridge kept weight 1: Q = 0.357.)
        assert!((result.modularity - 0.5).abs() < 1e-2, "{result:?}");
    }

    #[test]
    fn single_cluster_for_complete_graph() {
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
            (2, 3, 0.1),
        ];
        let g = make_graph(4, &edges);
        let result = fast_community(&g, &FastCommunityOptions::default());
        assert_eq!(result.n_communities, 1, "K4 should be one cluster");
    }

    #[test]
    fn labels_are_contiguous() {
        let edges = vec![(0, 1, 0.1), (2, 3, 0.1), (4, 5, 0.1)];
        let g = make_graph(6, &edges);
        let result = fast_community(&g, &FastCommunityOptions::default());
        let max_label = *result.labels.iter().max().unwrap();
        assert_eq!(max_label as usize + 1, result.n_communities);
    }

    #[test]
    fn deterministic_for_seed() {
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (1, 2, 0.1),
            (3, 4, 0.1),
            (3, 5, 0.1),
            (4, 5, 0.1),
            (2, 3, 5.0),
        ];
        let g = make_graph(6, &edges);
        let a = fast_community(
            &g,
            &FastCommunityOptions {
                seed: 5,
                ..Default::default()
            },
        );
        let b = fast_community(
            &g,
            &FastCommunityOptions {
                seed: 5,
                ..Default::default()
            },
        );
        assert_eq!(a.labels, b.labels);
    }
}
