#![doc = include_str!("../../README.md")]

mod repulsion;
mod tsne;

pub mod affinities;
pub mod error;
pub mod fit;

#[cfg(feature = "csv")]
mod csv;

#[cfg(feature = "csv")]
pub use csv::load_csv;

/// External re-exports required for consumers to name generic bounds and dimensionality markers
/// used by the crate.
pub use {
    barnes_hut_tree::{Dim, Morton},
    rustfft::FftNum,
    tsne::interpolation::FftDim,
    tsne::spectral::{SpectralBlock, SpectralParams},
};

/// Affinity type and builder re-exports.
pub use affinities::{Affinities, AffinitiesBuilder, Neighbor};

/// Error type re-exports.
pub use error::FromCsrError;

/// Fit builder and result re-exports.
pub use fit::{
    BhtsneBuilder, ExactBuilder, FitSneBuilder, FittedBhtsne, FittedExact, FittedFitSne,
    TsneBuilder, TsneReady,
};

/// Callback signature accepted by [`TsneBuilder::epoch_callback`]. Consumes the epoch index and
/// a snapshot of the current embedding.
pub type EpochCallback<'d, T> = Box<dyn FnMut(usize, &[T]) + 'd>;

/// Chunk size below which rayon falls back to a serial split. Used across the internal fit and
/// gradient kernels.
pub(crate) const PARALLEL_CODE_THRESHOLD: usize = 4096;

#[cfg(test)]
mod test;
