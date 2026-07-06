//! Exact t-SNE builder and result.

use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, DivAssign, MulAssign, SubAssign};

use num_traits::{Float, cast::AsPrimitive};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::FitConfig;
use crate::affinities::Affinities;
use crate::tsne;

/// Exact t-SNE builder.
///
/// Produced by [`TsneReady::exact`](crate::TsneReady::exact).
pub struct ExactBuilder<'d, T, U, const D: usize> {
    pub(super) config: FitConfig<'d, T, U, D>,
    pub(super) affinities: Affinities<T>,
}

/// Result of an exact t-SNE fit.
pub struct FittedExact<'d, T, U, const D: usize> {
    embedding: Vec<T>,
    kl_divergence: T,
    affinities: Affinities<T>,
    _marker: PhantomData<&'d U>,
}

impl<'d, T, U, const D: usize> ExactBuilder<'d, T, U, D>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign
        + Add,
    U: Send + Sync,
{
    /// Runs the exact t-SNE optimization loop and returns the fitted result.
    ///
    /// Densifies the sparse affinity graph into a full `n * n` P matrix,
    /// then runs the dense pairwise optimization.
    pub fn fit(mut self) -> FittedExact<'d, T, U, D> {
        let n_samples = self.config._data.len();
        let grad_entries = n_samples * D;
        let pairwise_entries = n_samples * n_samples;

        // Densify sparse P into a full n*n matrix
        let mut p_values = vec![T::zero(); pairwise_entries];
        let p_rows = self.affinities.rows();
        let p_columns = self.affinities.columns();
        let p_values_sparse = self.affinities.values();

        for (i, value) in p_values_sparse.iter().enumerate() {
            let row = find_row_for_entry(p_rows, i);
            let col = p_columns[i] as usize;
            p_values[row * n_samples + col] = *value;
        }

        let mut y = vec![T::zero(); grad_entries];
        let mut dy = vec![T::zero(); grad_entries];
        let mut uy = vec![T::zero(); grad_entries];
        let mut gains = vec![T::one(); grad_entries];
        let mut distances = vec![T::zero(); pairwise_entries];

        // Zero the diagonal
        for d in distances.iter_mut().step_by(n_samples + 1) {
            *d = T::zero();
        }

        // Normalize P with early exaggeration
        tsne::normalize_p_values(&mut p_values, self.config.early_exaggeration);

        // Stop lying immediately if stop_lying_epoch == 0
        if self.config.stop_lying_epoch == 0 {
            tsne::stop_lying(&mut p_values, self.config.early_exaggeration);
        }

        // Seed embedding
        match self.config.initial_embedding {
            Some(init) => {
                assert_eq!(
                    init.len(),
                    grad_entries,
                    "error: initial embedding has {} values, expected n_samples * D = {}",
                    init.len(),
                    grad_entries
                );
                y.copy_from_slice(&init);
            }
            None => match self.config.spectral_init {
                Some((params, seeder)) => {
                    // For exact, build a temporary CSR from dense P for spectral seeding
                    let (tmp_rows, tmp_cols, tmp_vals) = densify_to_csr(&p_values, n_samples);
                    let seed = seeder(&tmp_rows, &tmp_cols, &tmp_vals, params);
                    y.copy_from_slice(&seed);
                }
                None => tsne::random_init(&mut y),
            },
        }

        let learning_rate = self.config.learning_rate.unwrap_or_else(|| {
            let auto = T::from(n_samples).unwrap()
                / self.config.early_exaggeration
                / T::from(4.0).unwrap();
            auto.max(T::from(50.0).unwrap())
        });

        let (mut epoch_callback, mut snapshot) = match self.config.epoch_callback.take() {
            Some(cb) => (Some(cb), vec![T::zero(); grad_entries]),
            None => (None, Vec::new()),
        };

        let mut momentum = self.config.momentum;
        let mut stop_lying_fired = false;

        for epoch in 0..self.config.epochs {
            // Compute pairwise squared Euclidean distances in embedding space
            let (y_chunks, _) = y.as_chunks::<D>();
            tsne::compute_pairwise_distance_matrix(
                &mut distances,
                |ith: &[T; D], jth: &[T; D]| {
                    ith.iter()
                        .zip(jth.iter())
                        .map(|(&i, &j)| (i - j).powi(2))
                        .sum()
                },
                |index| &y_chunks[*index],
                n_samples,
            );

            // Compute Q
            let mut q_values = vec![T::zero(); pairwise_entries];
            for (q, d) in q_values.iter_mut().zip(distances.iter()) {
                *q = (T::one() + *d).recip();
            }

            // Compute exact gradient
            let q_values_sum: T = q_values.par_iter().copied().sum();
            let inverse_q_sum = q_values_sum.recip();

            for sample_i in 0..n_samples {
                let dy_sample = &mut dy[sample_i * D..(sample_i + 1) * D];
                let y_sample = &y[sample_i * D..(sample_i + 1) * D];
                let p_sample = &p_values[sample_i * n_samples..(sample_i + 1) * n_samples];
                let q_sample = &q_values[sample_i * n_samples..(sample_i + 1) * n_samples];

                for sample_j in 0..n_samples {
                    let p = p_sample[sample_j];
                    let q = q_sample[sample_j];
                    let m = (p - q * inverse_q_sum) * q;
                    let y_j = &y[sample_j * D..(sample_j + 1) * D];

                    for d in 0..D {
                        dy_sample[d] += (y_sample[d] - y_j[d]) * m;
                    }
                }
            }

            // Update embedding
            tsne::update_solution(&mut y, &dy, &mut uy, &mut gains, &learning_rate, &momentum);

            dy.fill(T::zero());
            tsne::zero_mean::<T, D>(&mut y, n_samples);

            // Epoch tail
            if epoch == self.config.stop_lying_epoch && epoch != 0 && !stop_lying_fired {
                tsne::stop_lying(&mut p_values, self.config.early_exaggeration);
                stop_lying_fired = true;
            }
            if epoch == self.config.momentum_switch_epoch {
                momentum = self.config.final_momentum;
            }
            if let Some(ref mut callback) = epoch_callback {
                snapshot.copy_from_slice(&y);
                callback(epoch, &snapshot);
            }
        }

        let kl_divergence = tsne::evaluate_error::<T, D>(&p_values, &y, n_samples);

        FittedExact {
            embedding: y,
            kl_divergence,
            affinities: self.affinities,
            _marker: PhantomData,
        }
    }
}

impl<'d, T, U, const D: usize> FittedExact<'d, T, U, D> {
    /// Returns the final embedding as a row-major slice of length `n_samples * D`.
    pub fn embedding(&self) -> &[T] {
        &self.embedding
    }

    /// Returns the final Kullback-Leibler divergence.
    pub fn kl_divergence(&self) -> T
    where
        T: Copy,
    {
        self.kl_divergence
    }

    /// Returns the affinity graph used for the fit.
    pub fn affinities(&self) -> &Affinities<T> {
        &self.affinities
    }
}

/// Given a flat index into the columns array, find which row it belongs to.
fn find_row_for_entry(rows: &[usize], index: usize) -> usize {
    // Binary search: find the largest row i such that rows[i] <= index
    let mut lo = 0usize;
    let mut hi = rows.len() - 1;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if rows[mid] <= index {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Convert a dense P matrix to CSR form for spectral seeding.
fn densify_to_csr<T: Float + Copy>(
    p_values: &[T],
    n_samples: usize,
) -> (Vec<usize>, Vec<u32>, Vec<T>) {
    let mut rows = vec![0usize; n_samples + 1];
    let mut nnz = 0usize;
    for i in 0..n_samples {
        for j in 0..n_samples {
            if p_values[i * n_samples + j] != T::zero() {
                nnz += 1;
            }
        }
        rows[i + 1] = nnz;
    }
    let mut columns = vec![0u32; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut current = vec![0usize; n_samples];
    for i in 0..n_samples {
        for j in 0..n_samples {
            let v = p_values[i * n_samples + j];
            if v != T::zero() {
                let pos = rows[i] + current[i];
                columns[pos] = j as u32;
                values[pos] = v;
                current[i] += 1;
            }
        }
    }
    (rows, columns, values)
}
