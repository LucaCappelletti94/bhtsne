//! Error types for fallible bhtsne APIs.

use thiserror::Error;

/// Errors from [`Affinities::from_csr`].
///
/// [`Affinities::from_csr`]: crate::Affinities::from_csr
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FromCsrError {
    #[error("from_csr: rows has length {actual}, expected n_samples + 1 = {expected}")]
    WrongRowsLen { expected: usize, actual: usize },
    #[error("from_csr: rows[{at}] = {next} is less than rows[{prev_index}] = {prev}",
            prev_index = at - 1)]
    NonMonotoneRows { at: usize, prev: usize, next: usize },
    #[error("from_csr: rows[n] = {rows_end} does not match columns.len() = {columns_len}")]
    RowsEndMismatch { rows_end: usize, columns_len: usize },
    #[error("from_csr: row {row} entry {entry} has column {column} out of range 0..{n_samples}")]
    ColumnOutOfRange {
        row: usize,
        entry: usize,
        column: u32,
        n_samples: usize,
    },
    #[error("from_csr: value at row {row} entry {entry} is not finite")]
    ValueNotFinite { row: usize, entry: usize },
    #[error("from_csr: value at row {row} entry {entry} is negative")]
    ValueNegative { row: usize, entry: usize },
    #[error("from_csr: graph is not symmetric at (row {row}, col {col})")]
    Asymmetric { row: usize, col: usize },
    #[error("from_csr: all values sum to zero; graph cannot be normalised")]
    ZeroTotalMass,
}

/// Errors from [`AffinitiesBuilder`] methods.
///
/// [`AffinitiesBuilder`]: crate::AffinitiesBuilder
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AffinitiesBuilderError {
    #[error("view weight must be strictly positive")]
    NonPositiveWeight,
    #[error("view spans {view} samples, expected the builder's {builder}")]
    SampleCountMismatch { view: usize, builder: usize },
    #[error("subset view spans {view} samples, but subset has {subset} ids")]
    SubsetSizeMismatch { view: usize, subset: usize },
    #[error("subset id {id} is out of range 0..{n_samples}")]
    SubsetIdOutOfRange { id: usize, n_samples: usize },
    #[error("subset has duplicate global id {id}")]
    DuplicateSubsetId { id: usize },
    #[error("agreement temperature tau must be strictly positive")]
    NonPositiveTau,
    #[error("builder needs at least one view before building")]
    NoViews,
}

/// Errors from the PCA-required metric sugar constructors [`Affinities::from_l2`] and
/// [`Affinities::from_cosine`].
///
/// [`Affinities::from_l2`]: crate::Affinities::from_l2
/// [`Affinities::from_cosine`]: crate::Affinities::from_cosine
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PcaError {
    #[error("no rows supplied")]
    NoRows,
    #[error("target_dim must be at least 1")]
    ZeroTargetDim,
    #[error("row {row} has {actual} entries, expected {expected}")]
    RaggedRows {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("perplexity ~{perplexity} is too large for {n} samples; need 3 * perplexity < n")]
    PerplexityTooLarge { perplexity: usize, n: usize },
}

/// Errors from [`Affinities::from_shortest_path`].
///
/// [`Affinities::from_shortest_path`]: crate::Affinities::from_shortest_path
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FromShortestPathError {
    #[error("edge endpoint {id} is out of range 0..{n_samples}")]
    EdgeIdOutOfRange { id: usize, n_samples: usize },
    #[error("perplexity ~{perplexity} is too large for {n} samples; need 3 * perplexity < n")]
    PerplexityTooLarge { perplexity: usize, n: usize },
}
