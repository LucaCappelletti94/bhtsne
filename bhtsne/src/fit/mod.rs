//! Type-state builders for t-SNE fitting.
//!
//! The entry point is [`TsneBuilder::new`], which configures the fit hyperparameters,
//! then transitions through [`TsneReady`] (after injecting a pre-built [`Affinities`]
//! graph) to one of the strategy-specific builders.
//!
//! [`Affinities`]: crate::affinities::Affinities

mod bhtsne;
mod exact;
mod fit_sne;

pub use bhtsne::{BhtsneBuilder, FittedBhtsne};
pub use exact::{ExactBuilder, FittedExact};
pub use fit_sne::{FitSneBuilder, FittedFitSne};

use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use barnes_hut_tree::Dim;
use num_traits::{Float, cast::AsPrimitive};

use crate::affinities::Affinities;
use crate::tsne::spectral::{SpectralBlock, SpectralParams};

/// Monomorphized spectral solver captured by [`TsneBuilder::spectral_init_with`],
/// where the [`SpectralBlock`] bound is available, and invoked by the seeding step
/// of the fit, where it is not.
type SpectralSeeder<T> = fn(&[usize], &[u32], &[T], SpectralParams) -> Vec<T>;

/// Private configuration shared across the builder states.
struct FitConfig<'d, T, U, const D: usize> {
    _data: &'d [U],
    _phantom: PhantomData<T>,
    learning_rate: Option<T>,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    early_exaggeration: T,
    initial_embedding: Option<Vec<T>>,
    spectral_init: Option<(SpectralParams, SpectralSeeder<T>)>,
    epoch_callback: Option<crate::EpochCallback<'d, T>>,
}

/// Type-state builder for t-SNE fit configuration.
///
/// Configure hyperparameters via the consuming setter methods, then call
/// [`with_affinities`](Self::with_affinities) to inject a pre-built affinity graph
/// and transition to [`TsneReady`].
pub struct TsneBuilder<'d, T, U, const D: usize = 2> {
    config: FitConfig<'d, T, U, D>,
}

/// Affinity-loaded state: no more config setters, only strategy transitions.
///
/// Choose a repulsion strategy via [`bhtsne`](Self::bhtsne), [`fit_sne`](Self::fit_sne),
/// or [`exact`](Self::exact).
pub struct TsneReady<'d, T, U, const D: usize> {
    config: FitConfig<'d, T, U, D>,
    affinities: Affinities<T>,
}

impl<'d, T, U, const D: usize> TsneBuilder<'d, T, U, D>
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
{
    /// Creates a new t-SNE fit builder for the given dataset.
    ///
    /// Default configuration:
    /// - `learning_rate` = auto (`max(n_samples / early_exaggeration / 4, 50)`)
    /// - `epochs` = 1000
    /// - `momentum` = 0.5
    /// - `final_momentum` = 0.8
    /// - `momentum_switch_epoch` = 250
    /// - `stop_lying_epoch` = 250
    /// - `early_exaggeration` = 12.0
    pub fn new(data: &'d [U]) -> Self {
        Self {
            config: FitConfig {
                _data: data,
                _phantom: PhantomData,
                learning_rate: None,
                epochs: 1000,
                momentum: T::from(0.5).unwrap(),
                final_momentum: T::from(0.8).unwrap(),
                momentum_switch_epoch: 250,
                stop_lying_epoch: 250,
                early_exaggeration: T::from(12.0).unwrap(),
                initial_embedding: None,
                spectral_init: None,
                epoch_callback: None,
            },
        }
    }

    /// Sets an explicit learning rate, overriding the size-scaled default.
    pub fn learning_rate(mut self, learning_rate: T) -> Self {
        self.config.learning_rate = Some(learning_rate);
        self
    }

    /// Sets the maximum number of fitting iterations.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// Sets the initial momentum coefficient.
    pub fn momentum(mut self, momentum: T) -> Self {
        self.config.momentum = momentum;
        self
    }

    /// Sets the momentum coefficient used after [`momentum_switch_epoch`](Self::momentum_switch_epoch).
    pub fn final_momentum(mut self, final_momentum: T) -> Self {
        self.config.final_momentum = final_momentum;
        self
    }

    /// Sets the epoch after which momentum switches to `final_momentum`.
    pub fn momentum_switch_epoch(mut self, momentum_switch_epoch: usize) -> Self {
        self.config.momentum_switch_epoch = momentum_switch_epoch;
        self
    }

    /// Sets the epoch after which the P distribution values become true.
    ///
    /// For epochs before `stop_lying_epoch`, the P distribution values are
    /// multiplied by the `early_exaggeration` factor. A value of `0` disables
    /// early exaggeration entirely.
    pub fn stop_lying_epoch(mut self, stop_lying_epoch: usize) -> Self {
        self.config.stop_lying_epoch = stop_lying_epoch;
        self
    }

    /// Sets the early exaggeration factor applied to the P distribution
    /// during the initial epochs.
    pub fn early_exaggeration(mut self, early_exaggeration: T) -> Self {
        self.config.early_exaggeration = early_exaggeration;
        self
    }

    /// Sets a callback invoked at the end of each fitting epoch.
    ///
    /// The callback receives the zero-based epoch index and a snapshot of the
    /// current embedding. It is invoked sequentially from the fitting thread.
    pub fn epoch_callback<C>(mut self, callback: C) -> Self
    where
        C: FnMut(usize, &[T]) + 'd,
    {
        self.config.epoch_callback = Some(Box::new(callback));
        self
    }

    /// Seeds the embedding with the given coordinates instead of random initialization.
    ///
    /// The seed is consumed by the next fit and must have length `n_samples * D`.
    pub fn initial_embedding(mut self, embedding: impl Into<Vec<T>>) -> Self {
        self.config.initial_embedding = Some(embedding.into());
        self
    }

    /// Use spectral embedding initialization with default [`SpectralParams`].
    pub fn spectral_init(mut self) -> Self
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        self.config.spectral_init = Some((
            SpectralParams::default(),
            crate::tsne::spectral::spectral_embedding::<T, D>,
        ));
        self
    }

    /// Use spectral embedding initialization with custom [`SpectralParams`].
    pub fn spectral_init_with(mut self, params: SpectralParams) -> Self
    where
        T: Default,
        Dim<D>: SpectralBlock,
    {
        self.config.spectral_init =
            Some((params, crate::tsne::spectral::spectral_embedding::<T, D>));
        self
    }

    /// Injects a pre-built affinity graph and transitions to [`TsneReady`].
    ///
    /// The perplexity travels with the affinity constructor, not the fit builder,
    /// so there is no `perplexity` setter on this builder.
    pub fn with_affinities(self, affinities: Affinities<T>) -> TsneReady<'d, T, U, D> {
        TsneReady {
            config: self.config,
            affinities,
        }
    }
}

impl<'d, T, U, const D: usize> TsneReady<'d, T, U, D>
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
{
    /// Transition to the Barnes-Hut t-SNE builder.
    ///
    /// # Panics
    ///
    /// If `theta <= 0.0`.
    pub fn bhtsne(self, theta: T) -> BhtsneBuilder<'d, T, U, D>
    where
        Dim<D>: barnes_hut_tree::Morton<D>,
    {
        assert!(
            theta > T::zero(),
            "error: theta value must be greater than 0.0.\n\
             A value of 0.0 corresponds to using the exact version of the algorithm."
        );
        BhtsneBuilder {
            config: self.config,
            affinities: self.affinities,
            theta,
        }
    }

    /// Transition to the FIt-SNE builder (FFT-interpolated repulsion).
    ///
    /// Restricted to `D in {1, 2}`.
    pub fn fit_sne(self) -> FitSneBuilder<'d, T, U, D>
    where
        T: rustfft::FftNum,
        Dim<D>: crate::FftDim,
    {
        FitSneBuilder {
            config: self.config,
            affinities: self.affinities,
        }
    }

    /// Transition to the exact t-SNE builder.
    pub fn exact(self) -> ExactBuilder<'d, T, U, D> {
        ExactBuilder {
            config: self.config,
            affinities: self.affinities,
        }
    }
}
