//! [`Affinities::from_edges`] and [`Affinities::from_weighted_edges`] constructors plus their
//! focused tests. Both share the same CSR-build body, so `from_edges` delegates to the weighted
//! variant with unit weights.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use super::Affinities;
use crate::tsne;

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
    /// Builds a normalized affinity graph from an iterator of undirected unit-weight edges.
    ///
    /// Each edge `(i, j)` creates a symmetric entry with unit weight. Duplicate edges are
    /// accumulated (their weights add). The resulting graph is normalized to sum to one.
    pub fn from_edges<I>(n_samples: usize, edges: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        Self::from_weighted_edges(n_samples, edges.into_iter().map(|(i, j)| (i, j, T::one())))
    }

    /// Builds a normalized affinity graph from an iterator of undirected weighted edges.
    ///
    /// Each edge `(i, j, w)` is treated as a symmetric affinity, contributing `w` to both
    /// `(i, j)` and `(j, i)`. Duplicate edges accumulate (their weights add). The resulting
    /// graph is normalized to sum to one. Weights are taken as raw affinities: no
    /// distance-to-similarity conversion and no perplexity calibration is applied. Non-finite
    /// or negative weights propagate through the sum and normalization unchanged, so callers
    /// that want valid t-SNE input must supply non-negative finite weights.
    ///
    /// # Panics
    ///
    /// If any edge endpoint is `>= n_samples`.
    pub fn from_weighted_edges<I>(n_samples: usize, edges: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let mut accumulators: Vec<Vec<(u32, T)>> = vec![Vec::new(); n_samples];

        for (i, j, w) in edges {
            if let Some(entry) = accumulators[i].iter_mut().find(|(c, _)| *c == j as u32) {
                entry.1 += w;
            } else {
                accumulators[i].push((j as u32, w));
            }
            if let Some(entry) = accumulators[j].iter_mut().find(|(c, _)| *c == i as u32) {
                entry.1 += w;
            } else {
                accumulators[j].push((i as u32, w));
            }
        }

        let mut rows: Vec<usize> = Vec::with_capacity(n_samples + 1);
        let mut columns: Vec<u32> = Vec::new();
        let mut values: Vec<T> = Vec::new();
        rows.push(0);
        for mut acc in accumulators {
            acc.sort_unstable_by_key(|(c, _)| *c);
            for (col, val) in acc {
                columns.push(col);
                values.push(val);
            }
            rows.push(columns.len());
        }

        tsne::normalize_p_values(&mut values, T::one());

        Self {
            rows,
            columns,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Affinities;

    /// `from_edges` compiles over an owned edge iterator and produces a symmetric normalized
    /// graph. Regression: an earlier version had a bogus `AsRef<(usize, usize)>` bound that made
    /// the natural caller shape uncallable.
    #[test]
    fn from_edges_builds_symmetric_normalized_graph() {
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4)];
        let graph = Affinities::<f32>::from_edges(5, edges.iter().copied());
        assert_eq!(graph.n_samples(), 5);
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "graph values sum {sum} != 1");
        // Every edge produces a symmetric pair. Endpoint rows have 1 entry, interior rows 2.
        let rows = graph.rows();
        assert_eq!(rows[1] - rows[0], 1);
        assert_eq!(rows[2] - rows[1], 2);
        assert_eq!(rows[3] - rows[2], 2);
        assert_eq!(rows[4] - rows[3], 2);
        assert_eq!(rows[5] - rows[4], 1);
        // Also compiles with an owned array passed by value.
        let owned = [(0usize, 1usize), (1, 2)];
        let graph2 = Affinities::<f32>::from_edges(3, owned);
        assert_eq!(graph2.n_samples(), 3);
    }

    /// `from_weighted_edges` respects the supplied weights and matches `from_edges` when every
    /// weight is one. Verifies symmetric CSR layout, normalization, duplicate accumulation, and
    /// weight proportionality across a two-edge graph.
    #[test]
    fn from_weighted_edges_respects_weights_and_normalizes() {
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4)];
        let unit_ref = Affinities::<f32>::from_edges(5, edges.iter().copied());
        let unit_via_weighted =
            Affinities::<f32>::from_weighted_edges(5, edges.iter().map(|&(i, j)| (i, j, 1.0f32)));
        assert_eq!(unit_ref.rows(), unit_via_weighted.rows());
        assert_eq!(unit_ref.columns(), unit_via_weighted.columns());
        for (a, b) in unit_ref
            .values()
            .iter()
            .zip(unit_via_weighted.values().iter())
        {
            assert!((a - b).abs() < 1e-6);
        }

        // Edge (0,1) with weight 3 must carry three times the share of (1,2) with weight 1
        // after normalization to sum-one.
        let weighted = [(0usize, 1usize, 3.0f32), (1, 2, 1.0)];
        let graph = Affinities::<f32>::from_weighted_edges(3, weighted);
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "values sum {sum} != 1");
        let row0_val = graph.values()[graph.rows()[0]];
        let row2_val = graph.values()[graph.rows()[2]];
        assert!(
            (row0_val / row2_val - 3.0).abs() < 1e-5,
            "weight ratio broke: {row0_val} vs {row2_val}"
        );

        // Duplicate edges accumulate: two (0,1,1.0) equal one (0,1,2.0).
        let dup = [(0usize, 1usize, 1.0f32), (0, 1, 1.0), (1, 2, 1.0)];
        let single = [(0usize, 1usize, 2.0f32), (1, 2, 1.0)];
        let g_dup = Affinities::<f32>::from_weighted_edges(3, dup);
        let g_single = Affinities::<f32>::from_weighted_edges(3, single);
        assert_eq!(g_dup.rows(), g_single.rows());
        assert_eq!(g_dup.columns(), g_single.columns());
        for (a, b) in g_dup.values().iter().zip(g_single.values().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
