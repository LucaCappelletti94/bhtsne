//! Barnes-Hut t-SNE builder and result.

use num_traits::{Float, cast::AsPrimitive};
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use super::FitConfig;
use crate::affinities::Affinities;
use crate::repulsion::{BarnesHutRepulsion, Repulsion};
use crate::tsne;

/// Barnes-Hut t-SNE builder.
///
/// Produced by [`TsneReady::bhtsne`](crate::TsneReady::bhtsne).
pub struct BhtsneBuilder<'d, T, U, const D: usize> {
    pub(super) config: FitConfig<'d, T, U, D>,
    pub(super) affinities: Affinities<T>,
    pub(super) theta: T,
}

/// Result of a Barnes-Hut t-SNE fit.
pub struct FittedBhtsne<'d, T, U, const D: usize> {
    embedding: Vec<T>,
    kl_divergence: T,
    affinities: Affinities<T>,
    _marker: PhantomData<&'d U>,
}

impl<'d, T, U, const D: usize> BhtsneBuilder<'d, T, U, D>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign,
    U: Send + Sync,
    barnes_hut_tree::Dim<D>: barnes_hut_tree::Morton<D>,
{
    /// Runs the Barnes-Hut t-SNE optimization loop and returns the fitted result.
    pub fn fit(mut self) -> FittedBhtsne<'d, T, U, D> {
        let n_samples = self.config._data.len();
        let grad_entries = n_samples * D;

        let p_rows = self.affinities.rows();
        let p_columns = self.affinities.columns();
        let mut p_values = self.affinities.values().to_vec();

        let mut y = vec![T::zero(); grad_entries];
        let mut uy = vec![T::zero(); grad_entries];
        let mut gains = vec![T::one(); grad_entries];
        let mut positive_forces = vec![T::zero(); grad_entries];
        let mut negative_forces = vec![T::zero(); grad_entries];

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
                    let seed = seeder(p_rows, p_columns, &p_values, params);
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

        let mut strategy = BarnesHutRepulsion::<
            T,
            <barnes_hut_tree::Dim<D> as barnes_hut_tree::Morton<D>>::Word,
            D,
        >::new(self.theta);

        for epoch in 0..self.config.epochs {
            let inverse_norm = strategy.step(
                &y,
                p_rows,
                p_columns,
                &p_values,
                &mut positive_forces,
                &mut negative_forces,
            );

            tsne::gradient_descent_step::<T, D>(
                &mut y,
                &positive_forces,
                &negative_forces,
                &mut uy,
                &mut gains,
                tsne::GradientStep {
                    learning_rate,
                    momentum,
                    inverse_norm,
                },
            );

            tsne::zero_mean::<T, D>(&mut y, n_samples);

            // Epoch tail: stop lying, momentum switch, callback
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

        let kl_divergence = tsne::evaluate_error_approximately::<T, D>(
            p_rows, p_columns, &p_values, &y, n_samples, self.theta,
        );

        FittedBhtsne {
            embedding: y,
            kl_divergence,
            affinities: self.affinities,
            _marker: PhantomData,
        }
    }
}

impl<'d, T, U, const D: usize> FittedBhtsne<'d, T, U, D> {
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
