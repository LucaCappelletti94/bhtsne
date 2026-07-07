//! [`Affinities::from_shortest_path`] constructor, its `bfs_top_k_neighbors` helper, and its
//! focused tests.

use std::{
    collections::VecDeque,
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{Affinities, Neighbor};
use crate::error::FromShortestPathError;

impl<T> Affinities<T>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + AddAssign
        + DivAssign
        + MulAssign
        + SubAssign,
{
    /// Builds a normalized affinity graph from an undirected edge list by taking each node's
    /// `3 * perplexity` nearest neighbors under shortest-path (hop) distance and delegating to
    /// [`from_neighbors`]. Each BFS is truncated as soon as `k = 3 * perplexity` distinct
    /// non-source nodes are reached, so total work is `O(n * k * avg_degree)` rather than a
    /// full `O(n * (n + edges))`. Nodes in a component smaller than `k` end up with fewer
    /// neighbors (ragged, honestly reported to the fit); a fully isolated node keeps an empty
    /// row and drifts under repulsion alone.
    ///
    /// # Errors
    ///
    /// Returns [`FromShortestPathError::EdgeIdOutOfRange`] if any edge endpoint is
    /// `>= n_samples`, or [`FromShortestPathError::PerplexityTooLarge`] if
    /// `3 * perplexity >= n_samples`.
    ///
    /// [`from_neighbors`]: Affinities::from_neighbors
    pub fn from_shortest_path<I>(
        n_samples: usize,
        edges: I,
        perplexity: f64,
    ) -> Result<Self, FromShortestPathError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let perplexity_int: usize = perplexity as usize;
        if n_samples == 0 || n_samples - 1 < 3 * perplexity_int {
            return Err(FromShortestPathError::PerplexityTooLarge {
                perplexity: perplexity_int,
                n: n_samples,
            });
        }
        let k: usize = (3.0 * perplexity) as usize;

        // Build the undirected adjacency (sorted, deduped), rejecting self-loops and validating
        // each endpoint against `n_samples`.
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_samples];
        for (i, j) in edges {
            if i >= n_samples {
                return Err(FromShortestPathError::EdgeIdOutOfRange { id: i, n_samples });
            }
            if j >= n_samples {
                return Err(FromShortestPathError::EdgeIdOutOfRange { id: j, n_samples });
            }
            if i != j {
                adjacency[i].push(j as u32);
                adjacency[j].push(i as u32);
            }
        }
        for row in &mut adjacency {
            row.sort_unstable();
            row.dedup();
        }

        // Each source runs a BFS with early termination. Newly discovered nodes push to a FIFO
        // queue, so we visit in strict hop order; as soon as `k` non-source nodes are reached
        // we bail, bounding the expansion at `k` regardless of graph size (greedoid).
        let neighbors: Vec<Vec<Neighbor<T>>> = (0..n_samples)
            .into_par_iter()
            .map(|source| bfs_top_k_neighbors(&adjacency, source, k))
            .collect();

        Ok(Self::from_neighbors(&neighbors, perplexity))
    }
}

/// BFS from `source` truncated to the first `k` non-source nodes reached, returning them as
/// `(index, hop-distance)` pairs in visit order (already ascending by hop). Used by
/// [`Affinities::from_shortest_path`]. Isolated sources return an empty vector.
fn bfs_top_k_neighbors<T>(adjacency: &[Vec<u32>], source: usize, k: usize) -> Vec<Neighbor<T>>
where
    T: Float,
{
    if k == 0 {
        return Vec::new();
    }
    let n = adjacency.len();
    let mut dist = vec![u32::MAX; n];
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut order: Vec<u32> = Vec::with_capacity(k);

    dist[source] = 0;
    queue.push_back(source as u32);
    'outer: while let Some(node) = queue.pop_front() {
        let d = dist[node as usize];
        for &next in &adjacency[node as usize] {
            if dist[next as usize] == u32::MAX {
                dist[next as usize] = d + 1;
                queue.push_back(next);
                order.push(next);
                if order.len() >= k {
                    break 'outer;
                }
            }
        }
    }

    order
        .into_iter()
        .map(|idx| Neighbor {
            index: idx as usize,
            distance: T::from(dist[idx as usize]).expect("hop count fits in T"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::Affinities;
    use crate::error::FromShortestPathError as Err;

    /// Five-node chain 0-1-2-3-4 with perplexity 1 (`k = 3`): each node's top-3 hop neighbors
    /// give a symmetric CSR whose values sum to one and whose per-row entry counts reflect the
    /// chain topology.
    #[test]
    fn from_shortest_path_chain_five_perplexity_one() {
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4)];
        let graph = Affinities::<f32>::from_shortest_path(5, edges, 1.0).unwrap();
        assert_eq!(graph.n_samples(), 5);
        let rows = graph.rows();
        // The center node picks up all four others in <= 2 hops (top-3 leaves one out).
        assert!(rows[3] - rows[2] >= 3);
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// A component of size less than `k + 1` produces a ragged row on the isolated side, and
    /// a fully isolated node produces an empty row.
    #[test]
    fn from_shortest_path_isolated_node() {
        // Chain 0-1-2-3-4 plus a floating node 5. Node 5 has no incident edges.
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4)];
        let graph = Affinities::<f32>::from_shortest_path(6, edges, 1.0).unwrap();
        assert_eq!(graph.n_samples(), 6);
        let rows = graph.rows();
        // Node 5 has no attractive edges.
        assert_eq!(rows[6] - rows[5], 0);
    }

    /// Two disjoint triangles: BFS on either side only sees its own component, so the CSR
    /// splits cleanly along the two clusters.
    #[test]
    fn from_shortest_path_disconnected_components() {
        let edges = [(0usize, 1usize), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let graph = Affinities::<f32>::from_shortest_path(6, edges, 1.0).unwrap();
        let rows = graph.rows();
        let cols = graph.columns();
        for node in 0..3usize {
            for &c in &cols[rows[node]..rows[node + 1]] {
                assert!(
                    (c as usize) < 3,
                    "node {node} reached the other component via {c}"
                );
            }
        }
        for node in 3..6usize {
            for &c in &cols[rows[node]..rows[node + 1]] {
                assert!(
                    (c as usize) >= 3,
                    "node {node} reached the other component via {c}"
                );
            }
        }
    }

    /// A star of N leaves around node 0 with perplexity 3 (`k = 9`) truncates each leaf's BFS at
    /// the ninth non-source node found. After symmetrization every leaf still connects back to
    /// the center.
    #[test]
    fn from_shortest_path_star_truncation() {
        const N: usize = 31;
        let edges: Vec<(usize, usize)> = (1..N).map(|i| (0usize, i)).collect();
        let graph = Affinities::<f32>::from_shortest_path(N, edges.iter().copied(), 3.0).unwrap();
        let rows = graph.rows();
        // Center connects to every leaf.
        assert_eq!(rows[1] - rows[0], N - 1);
        // Each leaf keeps at least the center.
        for leaf in 1..N {
            assert!(rows[leaf + 1] - rows[leaf] >= 1);
        }
    }

    /// Both `FromShortestPathError` variants fire on the matching bad input.
    #[test]
    fn from_shortest_path_rejects_bad_input() {
        // Perplexity too large for 5 samples: 3 * 2.0 = 6 >= 5.
        let e = Affinities::<f32>::from_shortest_path(5, [(0usize, 1usize)], 2.0)
            .err()
            .unwrap();
        assert!(matches!(e, Err::PerplexityTooLarge { .. }));

        // Endpoint out of range.
        let e = Affinities::<f32>::from_shortest_path(5, [(0usize, 9usize)], 1.0)
            .err()
            .unwrap();
        assert!(matches!(
            e,
            Err::EdgeIdOutOfRange {
                id: 9,
                n_samples: 5
            }
        ));
    }
}
