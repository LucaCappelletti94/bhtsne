//! [`Affinities::from_csr`] and [`Affinities::from_csr_unchecked`] constructors and their
//! focused tests.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use super::Affinities;
use crate::error::FromCsrError;

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
    ) -> Result<Self, FromCsrError> {
        let expected = n_samples + 1;
        if rows.len() != expected {
            return Err(FromCsrError::WrongRowsLen {
                expected,
                actual: rows.len(),
            });
        }

        for i in 1..rows.len() {
            if rows[i] < rows[i - 1] {
                return Err(FromCsrError::NonMonotoneRows {
                    at: i,
                    prev: rows[i - 1],
                    next: rows[i],
                });
            }
        }

        if rows[n_samples] != columns.len() {
            return Err(FromCsrError::RowsEndMismatch {
                rows_end: rows[n_samples],
                columns_len: columns.len(),
            });
        }

        for i in 0..n_samples {
            for e in rows[i]..rows[i + 1] {
                if columns[e] as usize >= n_samples {
                    return Err(FromCsrError::ColumnOutOfRange {
                        row: i,
                        entry: e - rows[i],
                        column: columns[e],
                        n_samples,
                    });
                }
                if !values[e].is_finite() {
                    return Err(FromCsrError::ValueNotFinite {
                        row: i,
                        entry: e - rows[i],
                    });
                }
                if values[e] < T::zero() {
                    return Err(FromCsrError::ValueNegative {
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
                    return Err(FromCsrError::Asymmetric { row: i, col: j });
                }
            }
        }

        let total: T = values.iter().copied().sum();
        if total == T::zero() {
            return Err(FromCsrError::ZeroTotalMass);
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
    /// The caller vouches that `rows` has length `n_samples + 1`, is monotonically
    /// non-decreasing, and `rows[n_samples]` equals `columns.len()`. All column indices must be
    /// in `0..n_samples`.
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
}

#[cfg(test)]
mod tests {
    use crate::Affinities;
    use crate::error::FromCsrError as Err;

    /// Symmetric 3-node chain round-trips through validation and gets normalized to sum-one
    /// with the same row and column layout the caller supplied.
    #[test]
    fn from_csr_validates_symmetric_input_and_normalizes() {
        let rows = vec![0usize, 1, 3, 4];
        let columns = vec![1u32, 0, 2, 1];
        let values = vec![1.0f32, 1.0, 1.0, 1.0];
        let graph = Affinities::<f32>::from_csr(3, rows.clone(), columns.clone(), values).unwrap();

        assert_eq!(graph.n_samples(), 3);
        assert_eq!(graph.rows(), rows.as_slice());
        assert_eq!(graph.columns(), columns.as_slice());
        let sum: f32 = graph.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for &v in graph.values() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    /// Every `FromCsrError` variant fires on the matching malformed input, so callers see a
    /// pinpointed diagnostic instead of a panic on downstream indexing.
    #[test]
    fn from_csr_rejects_each_error_variant() {
        let good_rows = vec![0usize, 1, 3, 4];
        let good_columns = vec![1u32, 0, 2, 1];
        let good_values = vec![1.0f32, 1.0, 1.0, 1.0];

        // WrongRowsLen.
        let e = Affinities::<f32>::from_csr(
            3,
            vec![0usize, 1, 3],
            good_columns.clone(),
            good_values.clone(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            e,
            Err::WrongRowsLen {
                expected: 4,
                actual: 3
            }
        ));

        // NonMonotoneRows.
        let e = Affinities::<f32>::from_csr(
            3,
            vec![0usize, 3, 1, 4],
            good_columns.clone(),
            good_values.clone(),
        )
        .err()
        .unwrap();
        assert!(matches!(e, Err::NonMonotoneRows { .. }));

        // RowsEndMismatch.
        let e = Affinities::<f32>::from_csr(
            3,
            vec![0usize, 1, 3, 5],
            good_columns.clone(),
            good_values.clone(),
        )
        .err()
        .unwrap();
        assert!(matches!(e, Err::RowsEndMismatch { .. }));

        // ColumnOutOfRange.
        let e = Affinities::<f32>::from_csr(
            3,
            good_rows.clone(),
            vec![9u32, 0, 2, 1],
            good_values.clone(),
        )
        .err()
        .unwrap();
        assert!(matches!(e, Err::ColumnOutOfRange { column: 9, .. }));

        // ValueNotFinite.
        let mut bad_values = good_values.clone();
        bad_values[0] = f32::NAN;
        let e = Affinities::<f32>::from_csr(3, good_rows.clone(), good_columns.clone(), bad_values)
            .err()
            .unwrap();
        assert!(matches!(e, Err::ValueNotFinite { .. }));

        // ValueNegative.
        let mut bad_values = good_values.clone();
        bad_values[0] = -1.0;
        let e = Affinities::<f32>::from_csr(3, good_rows.clone(), good_columns.clone(), bad_values)
            .err()
            .unwrap();
        assert!(matches!(e, Err::ValueNegative { .. }));

        // Asymmetric: (0, 1) with weight 1 but no (1, 0).
        let asym_rows = vec![0usize, 1, 2, 3];
        let asym_columns = vec![1u32, 2, 1];
        let asym_values = vec![1.0f32, 1.0, 1.0];
        let e = Affinities::<f32>::from_csr(3, asym_rows, asym_columns, asym_values)
            .err()
            .unwrap();
        assert!(matches!(e, Err::Asymmetric { .. }));

        // ZeroTotalMass: symmetric all-zero.
        let zero_values = vec![0.0f32; good_columns.len()];
        let e = Affinities::<f32>::from_csr(3, good_rows, good_columns, zero_values)
            .err()
            .unwrap();
        assert!(matches!(e, Err::ZeroTotalMass));
    }

    /// The unchecked path is a pure move: no validation, no normalization. The caller's exact
    /// triple must reappear as the graph's internal storage.
    #[test]
    fn from_csr_unchecked_returns_input_verbatim() {
        // Deliberately asymmetric and unnormalized.
        let rows = vec![0usize, 1, 2];
        let columns = vec![1u32, 0];
        let values = vec![3.0f32, 4.0];
        let graph =
            Affinities::<f32>::from_csr_unchecked(2, rows.clone(), columns.clone(), values.clone());
        assert_eq!(graph.n_samples(), 2);
        assert_eq!(graph.rows(), rows.as_slice());
        assert_eq!(graph.columns(), columns.as_slice());
        assert_eq!(graph.values(), values.as_slice());
    }
}
