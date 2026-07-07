//! [`Affinities::from_metric`] constructor and its focused tests.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

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
    /// Builds a normalized affinity graph from `data` under the distance function `metric`, tuning
    /// each point's Gaussian bandwidth to `perplexity`.
    ///
    /// The returned graph is symmetric and its values sum to one.
    ///
    /// # Panics
    ///
    /// If `perplexity` is too large for the number of samples.
    pub fn from_metric<U, F>(data: &[U], perplexity: f64, metric: F) -> Self
    where
        U: Send + Sync,
        F: Fn(&U, &U) -> T + Send + Sync,
    {
        let n_samples = data.len();
        let perplexity_t: T = T::from(perplexity).expect("perplexity fits in T");
        tsne::check_perplexity(&perplexity_t, &n_samples);

        let n_neighbors: usize = (3.0 * perplexity) as usize;
        let tree = tsne::vptree::VPTree::new(data, &metric);

        let pairwise_entries = n_samples * n_neighbors;
        let mut values: Vec<T> = vec![T::zero(); pairwise_entries];
        let mut p_columns: Vec<u32> = vec![0u32; pairwise_entries];
        let mut distances: Vec<T> = vec![T::zero(); pairwise_entries];

        values
            .par_chunks_mut(n_neighbors)
            .zip(distances.par_chunks_mut(n_neighbors))
            .zip(p_columns.par_chunks_mut(n_neighbors))
            .enumerate()
            .for_each_init(
                tsne::vptree::SearchScratch::default,
                |scratch, (index, ((values_row, distances_row), p_columns_row))| {
                    tree.search(
                        &data[index],
                        index,
                        n_neighbors + 1,
                        (p_columns_row, distances_row),
                        scratch,
                        &metric,
                    );
                    tsne::search_beta(values_row, distances_row, &perplexity_t);
                },
            );

        drop(distances);
        drop(tree);

        let mut rows: Vec<usize> = Vec::new();
        let mut columns: Vec<u32> = Vec::new();
        tsne::symmetrize_sparse_matrix(
            &mut rows,
            &mut columns,
            p_columns,
            &mut values,
            n_samples,
            &n_neighbors,
        );

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
    use crate::testing::{brute_force_neighbors, euclidean, lcg_samples};

    const D: usize = 4;
    const PERPLEXITY: f64 = 10.0;

    /// The standalone `from_metric` constructor produces a well-formed symmetric CSR whose
    /// values sum to one, whose column indices are all in range, and whose row structure agrees
    /// with the ragged `from_neighbors` path on the same exact neighbors (row counts only; the
    /// two symmetrize paths do not preserve intra-row column ordering).
    #[test]
    fn from_metric_produces_valid_symmetric_graph() {
        const N: usize = 200;
        let data = lcg_samples(N, D, 42);
        let samples: Vec<&[f32]> = data.chunks(D).collect();

        let direct = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
            euclidean(a, b)
        });

        assert_eq!(direct.n_samples(), N);
        let sum: f32 = direct.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "values sum {sum} != 1");
        for &c in direct.columns() {
            assert!((c as usize) < N, "column {c} out of range");
        }
        for &v in direct.values() {
            assert!(v.is_finite() && v >= 0.0);
        }

        // Row counts must match the ragged `from_neighbors` path on the same exact neighbors:
        // both symmetrize paths preserve per-row entry counts, even though intra-row column
        // order and the specific added-back edges can differ between the dense and ragged
        // symmetrizers.
        let neighbors = brute_force_neighbors(&samples, (3.0 * PERPLEXITY) as usize);
        let via_neighbors = Affinities::from_neighbors(&neighbors, PERPLEXITY);
        assert_eq!(direct.rows(), via_neighbors.rows());
    }
}
