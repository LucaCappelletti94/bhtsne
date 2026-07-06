//! Error types returned by the fallible bhtsne APIs.
//!
//! Most of the crate panics on invalid inputs (perplexity too large, empty view lists, subset ids
//! out of range) because those are programmer errors, not runtime conditions. The one place that
//! returns a real [`Result`] is [`Affinities::from_csr`], where the caller injects a graph they
//! preprocessed themselves and needs to surface validation failures cleanly.
//!
//! [`Affinities::from_csr`]: crate::Affinities::from_csr

use std::fmt;

/// Errors produced by [`Affinities::from_csr`] when the caller-supplied CSR triple fails
/// validation.
///
/// [`Affinities::from_csr`]: crate::Affinities::from_csr
#[derive(Debug, Clone, PartialEq)]
pub enum FromCsrError {
    /// `rows` had the wrong length; it must be exactly `n_samples + 1`.
    WrongRowsLen { expected: usize, actual: usize },
    /// `rows` was not monotonically non-decreasing at some index.
    NonMonotoneRows { at: usize, prev: usize, next: usize },
    /// The last entry of `rows` did not match `columns.len()`.
    RowsEndMismatch { rows_end: usize, columns_len: usize },
    /// A column index was out of range `0..n_samples`.
    ColumnOutOfRange {
        row: usize,
        entry: usize,
        column: u32,
        n_samples: usize,
    },
    /// A value was not finite (NaN or infinite).
    ValueNotFinite { row: usize, entry: usize },
    /// A value was negative; affinities must be non-negative.
    ValueNegative { row: usize, entry: usize },
    /// The graph was not symmetric within tolerance; the caller vouched for a symmetric joint P.
    Asymmetric { row: usize, col: usize },
    /// All values summed to zero (or effectively zero after clamping), so the graph cannot be
    /// normalised.
    ZeroTotalMass,
}

impl fmt::Display for FromCsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRowsLen { expected, actual } => write!(
                f,
                "from_csr: rows has length {actual}, expected n_samples + 1 = {expected}"
            ),
            Self::NonMonotoneRows { at, prev, next } => write!(
                f,
                "from_csr: rows[{at}] = {next} is less than rows[{}] = {prev}",
                at - 1
            ),
            Self::RowsEndMismatch {
                rows_end,
                columns_len,
            } => write!(
                f,
                "from_csr: rows[n] = {rows_end} does not match columns.len() = {columns_len}"
            ),
            Self::ColumnOutOfRange {
                row,
                entry,
                column,
                n_samples,
            } => write!(
                f,
                "from_csr: row {row} entry {entry} has column {column} \
                 which is out of range 0..{n_samples}"
            ),
            Self::ValueNotFinite { row, entry } => write!(
                f,
                "from_csr: value at row {row} entry {entry} is not finite"
            ),
            Self::ValueNegative { row, entry } => {
                write!(f, "from_csr: value at row {row} entry {entry} is negative")
            }
            Self::Asymmetric { row, col } => write!(
                f,
                "from_csr: graph is not symmetric at (row {row}, col {col})"
            ),
            Self::ZeroTotalMass => write!(
                f,
                "from_csr: all values sum to zero; graph cannot be normalised"
            ),
        }
    }
}

impl std::error::Error for FromCsrError {}
