//! [`Affinities::from_neighbors`] constructor and its focused tests.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{Affinities, Neighbor};
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
    /// Builds a normalized affinity graph from a caller-supplied nearest-neighbor table, tuning
    /// each point's Gaussian bandwidth to `perplexity`.
    ///
    /// Rows may have different lengths (ragged) and empty rows are allowed. An empty row means
    /// the caller has no neighbor evidence for that sample under this view, matching the
    /// missing-modality pattern; that sample contributes no attractive force in the resulting
    /// graph and drifts under repulsion alone during the fit unless another sample points at it
    /// through symmetrization. Non-empty rows calibrate their own beta over their own distances.
    /// The output is a symmetrized CSR graph whose values sum to one.
    ///
    /// # Panics
    ///
    /// If `neighbors` does not have one row per sample, a neighbor index is out of range, or
    /// `perplexity` is too large for the number of samples.
    pub fn from_neighbors(neighbors: &[Vec<Neighbor<T>>], perplexity: f64) -> Self {
        let n_samples = neighbors.len();
        for (i, row) in neighbors.iter().enumerate() {
            for n in row {
                assert!(
                    n.index < n_samples,
                    "error: neighbor index {} in row {i} is out of range 0..{n_samples}",
                    n.index
                );
            }
        }
        let perplexity_t: T = T::from(perplexity).expect("perplexity fits in T");
        tsne::check_perplexity(&perplexity_t, &n_samples);

        // Per-row: calibrate beta and collect (column, value) pairs. Empty rows skip the beta
        // search and stay empty in the output CSR.
        let row_results: Vec<(Vec<u32>, Vec<T>)> = neighbors
            .par_iter()
            .map(|row| {
                if row.is_empty() {
                    return (Vec::new(), Vec::new());
                }
                let k = row.len();
                let mut p_values = vec![T::zero(); k];
                let mut distances = vec![T::zero(); k];
                for (j, n) in row.iter().enumerate() {
                    distances[j] = n.distance;
                }
                tsne::search_beta(&mut p_values, &distances, &perplexity_t);

                let columns: Vec<u32> = row.iter().map(|n| n.index as u32).collect();
                (columns, p_values)
            })
            .collect();

        let mut rows: Vec<usize> = Vec::with_capacity(n_samples + 1);
        let mut columns: Vec<u32> = Vec::new();
        let mut values: Vec<T> = Vec::new();
        rows.push(0);
        for (cols, vals) in &row_results {
            columns.extend_from_slice(cols);
            values.extend_from_slice(vals);
            rows.push(columns.len());
        }

        let (rows, columns, mut values) = tsne::symmetrize_csr(&rows, &columns, &values);
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
    use crate::testing::{brute_force_neighbors, lcg_samples};

    const D: usize = 4;
    const PERPLEXITY: f64 = 10.0;

    /// Feeding the exact `3 * perplexity` neighbors produces a well-formed graph: correct
    /// n_samples, values sum to one, and every column index is in range.
    #[test]
    fn from_neighbors_builds_expected_graph() {
        const N: usize = 150;
        let data = lcg_samples(N, D, 7);
        let samples: Vec<&[f32]> = data.chunks(D).collect();
        let neighbors = brute_force_neighbors(&samples, (3.0 * PERPLEXITY) as usize);

        let graph = Affinities::from_neighbors(&neighbors, PERPLEXITY);
        assert_eq!(graph.n_samples(), N);
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "values sum {sum} != 1");
        for &c in graph.columns() {
            assert!((c as usize) < N);
        }
    }

    /// An empty row is allowed: the sample contributes no outgoing attractive edges and only
    /// receives whatever symmetrization brings in from other rows pointing at it.
    #[test]
    fn from_neighbors_allows_empty_rows() {
        const N: usize = 60;
        let data = lcg_samples(N, D, 41);
        let samples: Vec<&[f32]> = data.chunks(D).collect();
        let mut neighbors = brute_force_neighbors(&samples, (3.0 * PERPLEXITY) as usize);
        // Clear the last two rows to simulate a missing modality on two samples.
        neighbors[N - 1].clear();
        neighbors[N - 2].clear();

        let graph = Affinities::from_neighbors(&neighbors, PERPLEXITY);
        assert_eq!(graph.n_samples(), N);
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "values sum {sum} != 1");
    }
}
