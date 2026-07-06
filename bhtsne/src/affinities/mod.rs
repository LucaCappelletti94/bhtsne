//! Precomputed sparse affinity graphs for t-SNE fitting.
//!
//! An [`Affinities`] value is a symmetric, CSR-layout graph where the values sum to one. Build one
//! from raw data, precomputed neighbors, edges, or an external CSR triple, then inject it into the
//! fit builder.

mod builder;

pub use builder::AffinitiesBuilder;

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign},
};

use num_traits::{Float, cast::AsPrimitive};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::tsne;

/// A sample's nearest neighbor: its index and the distance (not a similarity) to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor<T> {
    /// Neighbor sample index.
    pub index: usize,
    /// Distance to the neighbor.
    pub distance: T,
}

/// A precomputed sparse, symmetric affinity graph in CSR layout.
///
/// The `rows` array has length `n_samples + 1`, `columns` holds the column indices as `u32`,
/// and `values` holds the non-negative affinity weights. The graph is symmetric and the values
/// sum to one.
#[derive(Clone)]
pub struct Affinities<T> {
    rows: Vec<usize>,
    columns: Vec<u32>,
    values: Vec<T>,
}

// ─── accessors (no trait bound) ───

impl<T> Affinities<T> {
    /// Number of samples the graph spans.
    pub fn n_samples(&self) -> usize {
        self.rows.len().saturating_sub(1)
    }

    /// CSR row offsets.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// CSR column indices.
    pub fn columns(&self) -> &[u32] {
        &self.columns
    }

    /// Affinity values.
    pub fn values(&self) -> &[T] {
        &self.values
    }
}

// ─── constructors (standard Float bound) ───

impl<T> Affinities<T>
where
    T: Float + Send + Sync + AsPrimitive<usize> + Sum + AddAssign + DivAssign + MulAssign,
{
    /// Builds a normalized affinity graph from `data` under the distance function `metric`, tuning
    /// each point's Gaussian bandwidth to `perplexity`.
    ///
    /// The returned graph is symmetric and its values sum to one.
    ///
    /// # Panics
    ///
    /// If `perplexity` is too large for the number of samples.
    pub fn from_metric<U, F>(data: &[U], perplexity: T, metric: F) -> Self
    where
        U: Send + Sync,
        F: Fn(&U, &U) -> T + Send + Sync,
    {
        let n_samples = data.len();
        tsne::check_perplexity(&perplexity, &n_samples);

        let n_neighbors: usize = (T::from(3.0).unwrap() * perplexity).as_();
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
                    tsne::search_beta(values_row, distances_row, &perplexity);
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

    /// Builds a normalized affinity graph from a caller-supplied nearest-neighbor table, tuning
    /// each point's Gaussian bandwidth to `perplexity`.
    ///
    /// Rows may have different lengths (ragged) and empty rows are allowed. An empty row means the
    /// caller has no neighbor evidence for that sample under this view, matching the
    /// missing-modality pattern; that sample contributes no attractive force in the resulting
    /// graph and drifts under repulsion alone during the fit unless another sample points at it
    /// through symmetrization. Non-empty rows calibrate their own beta over their own distances.
    /// The output is a symmetrized CSR graph whose values sum to one.
    ///
    /// # Panics
    ///
    /// If `neighbors` does not have one row per sample, a neighbor index is out of range, or
    /// `perplexity` is too large for the number of samples.
    pub fn from_neighbors(neighbors: &[Vec<Neighbor<T>>], perplexity: T) -> Self {
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
        tsne::check_perplexity(&perplexity, &n_samples);

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
                tsne::search_beta(&mut p_values, &distances, &perplexity);

                let columns: Vec<u32> = row.iter().map(|n| n.index as u32).collect();
                (columns, p_values)
            })
            .collect();

        // Build CSR from the ragged per-row results.
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

    /// Builds a normalized affinity graph from an iterator of undirected edges.
    ///
    /// Each edge `(i, j)` creates a symmetric entry with unit weight. The resulting graph is
    /// normalized to sum to one. Duplicate edges are accumulated (their weights add).
    pub fn from_edges<I>(n_samples: usize, edges: I) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<(usize, usize)>,
    {
        let mut accumulators: Vec<Vec<(u32, T)>> = vec![Vec::new(); n_samples];

        for edge in edges {
            let (i, j) = *edge.as_ref();
            let one = T::one();
            if let Some(entry) = accumulators[i].iter_mut().find(|(c, _)| *c == j as u32) {
                entry.1 += one;
            } else {
                accumulators[i].push((j as u32, one));
            }
            if let Some(entry) = accumulators[j].iter_mut().find(|(c, _)| *c == i as u32) {
                entry.1 += one;
            } else {
                accumulators[j].push((i as u32, one));
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

    /// Builds an [`Affinities`] from a caller-supplied CSR triple with full validation.
    ///
    /// Validates: `rows` length is `n_samples + 1`, `rows` is monotonically non-decreasing,
    /// `rows[n_samples]` equals `columns.len()`, all column indices are in `0..n_samples`,
    /// all values are finite and non-negative, the graph is symmetric within tolerance
    /// `T::from(1e-6).unwrap()`, and the total mass is non-zero. On success the values are
    /// normalized to sum to one.
    pub fn from_csr(
        n_samples: usize,
        rows: Vec<usize>,
        columns: Vec<u32>,
        mut values: Vec<T>,
    ) -> Result<Self, crate::error::FromCsrError> {
        let expected = n_samples + 1;
        if rows.len() != expected {
            return Err(crate::error::FromCsrError::WrongRowsLen {
                expected,
                actual: rows.len(),
            });
        }

        for i in 1..rows.len() {
            if rows[i] < rows[i - 1] {
                return Err(crate::error::FromCsrError::NonMonotoneRows {
                    at: i,
                    prev: rows[i - 1],
                    next: rows[i],
                });
            }
        }

        if rows[n_samples] != columns.len() {
            return Err(crate::error::FromCsrError::RowsEndMismatch {
                rows_end: rows[n_samples],
                columns_len: columns.len(),
            });
        }

        for i in 0..n_samples {
            for e in rows[i]..rows[i + 1] {
                if columns[e] as usize >= n_samples {
                    return Err(crate::error::FromCsrError::ColumnOutOfRange {
                        row: i,
                        entry: e - rows[i],
                        column: columns[e],
                        n_samples,
                    });
                }
                if !values[e].is_finite() {
                    return Err(crate::error::FromCsrError::ValueNotFinite {
                        row: i,
                        entry: e - rows[i],
                    });
                }
                if values[e] < T::zero() {
                    return Err(crate::error::FromCsrError::ValueNegative {
                        row: i,
                        entry: e - rows[i],
                    });
                }
            }
        }

        // Symmetry check: for every edge (i, j, v) verify (j, i, v) exists.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            for e in rows[i]..rows[i + 1] {
                let j = columns[e] as usize;
                let v = values[e];
                let found = (rows[j]..rows[j + 1]).any(|f| {
                    columns[f] == i as u32 && (values[f] - v).abs() <= T::from(1e-6).unwrap()
                });
                if !found {
                    return Err(crate::error::FromCsrError::Asymmetric { row: i, col: j });
                }
            }
        }

        let total: T = values.iter().copied().sum();
        if total == T::zero() {
            return Err(crate::error::FromCsrError::ZeroTotalMass);
        }

        let scale = T::one() / total;
        for v in &mut values {
            *v *= scale;
        }

        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Builds an [`Affinities`] from a CSR triple with no validation or normalization.
    ///
    /// The caller vouches that `rows` has length `n_samples + 1`, is monotonically non-decreasing,
    /// and `rows[n_samples]` equals `columns.len()`. All column indices must be in `0..n_samples`.
    pub fn from_csr_unchecked(
        n_samples: usize,
        rows: Vec<usize>,
        columns: Vec<u32>,
        values: Vec<T>,
    ) -> Self {
        let _ = n_samples;
        Self {
            rows,
            columns,
            values,
        }
    }

    /// Sugar for [`from_metric`] with the Euclidean (L2) distance.
    ///
    /// [`from_metric`]: Affinities::from_metric
    pub fn from_l2(rows: &[&[T]], perplexity: T) -> Self {
        Self::from_metric(rows, perplexity, |a: &&[T], b: &&[T]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x - *y).powi(2))
                .sum::<T>()
                .sqrt()
        })
    }

    /// Sugar for [`from_metric`] with chord cosine distance.
    ///
    /// The caller vouches that each row is already L2-normalized; this function does not
    /// re-normalize.
    ///
    /// [`from_metric`]: Affinities::from_metric
    pub fn from_cosine(rows: &[&[T]], perplexity: T) -> Self {
        Self::from_metric(rows, perplexity, |a: &&[T], b: &&[T]| {
            let dot: T = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
            ((T::one() + T::one()) * (T::one() - dot).max(T::zero())).sqrt()
        })
    }
}
