//! [`Affinities::from_l2`] and [`Affinities::from_cosine`] PCA-reduction sugar constructors,
//! their shared `from_pca_metric` implementation, and focused tests.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use super::Affinities;
use crate::error::PcaError;

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
    /// Euclidean-metric sugar. Reduces `rows` to `target_dim` dimensions via mean-centered PCA,
    /// then builds the affinity graph under L2 distance on the reduced representation.
    ///
    /// # Errors
    ///
    /// Returns [`PcaError`] variants on any validation failure (empty input, ragged rows,
    /// `target_dim` out of range, perplexity too large for the sample count).
    pub fn from_l2(rows: &[&[T]], target_dim: usize, perplexity: f64) -> Result<Self, PcaError>
    where
        T: Default + Copy + MulAssign,
    {
        Self::from_pca_metric(rows, target_dim, perplexity, true, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x - *y).powi(2))
                .sum::<T>()
                .sqrt()
        })
    }

    /// Chord cosine sugar. Reduces `rows` to `target_dim` dimensions via uncentered PCA and
    /// evaluates chord cosine on the reduced rows. Truncation shrinks row norms, but the
    /// ordering the perplexity search cares about is preserved to within run-to-run kNN noise,
    /// so no post-hoc renormalization is applied.
    ///
    /// # Errors
    ///
    /// Returns [`PcaError`] variants on any validation failure.
    pub fn from_cosine(rows: &[&[T]], target_dim: usize, perplexity: f64) -> Result<Self, PcaError>
    where
        T: Default + Copy + MulAssign,
    {
        Self::from_pca_metric(rows, target_dim, perplexity, false, |a, b| {
            let dot: T = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
            ((T::one() + T::one()) * (T::one() - dot).max(T::zero())).sqrt()
        })
    }

    /// Shared implementation for [`Affinities::from_l2`] and [`Affinities::from_cosine`]:
    /// PCA-reduce, sanity-check perplexity against the reduced sample count, and then run the
    /// metric-based affinity build on the reduced representation.
    fn from_pca_metric<F>(
        rows: &[&[T]],
        target_dim: usize,
        perplexity: f64,
        center: bool,
        metric: F,
    ) -> Result<Self, PcaError>
    where
        T: Default + Copy + MulAssign,
        F: Fn(&&[T], &&[T]) -> T + Send + Sync,
    {
        let reduced = crate::tsne::pca::pca_reduce(rows, target_dim, center)?;
        let n = rows.len();
        let perplexity_int: usize = perplexity as usize;
        if n <= 3 * perplexity_int {
            return Err(PcaError::PerplexityTooLarge {
                perplexity: perplexity_int,
                n,
            });
        }
        // `pca_reduce` passes through when `target_dim >= d`, so the reduced buffer is chunked
        // by the smaller of the requested target and the raw feature dim.
        let effective_dim = target_dim.min(rows.first().map_or(target_dim, |r| r.len()));
        let reduced_rows: Vec<&[T]> = reduced.chunks_exact(effective_dim).collect();
        Ok(Self::from_metric(&reduced_rows, perplexity, metric))
    }
}

#[cfg(test)]
mod tests {
    use crate::Affinities;
    use crate::error::PcaError as Err;

    /// Two well-separated Euclidean blobs in 40 dimensions reduced to 3 via centered PCA. Every
    /// sample's stored neighbors should overwhelmingly live in its own cluster.
    #[test]
    fn from_l2_recovers_cluster_structure() {
        const N: usize = 60;
        const D: usize = 40;
        let mut raw = vec![0.0f32; N * D];
        for i in 0..N {
            let cluster = if i < N / 2 { 0 } else { 1 };
            let jitter = ((i as f32) * 0.31).sin() * 0.05;
            let row = &mut raw[i * D..(i + 1) * D];
            let anchor = if cluster == 0 { 0 } else { 20 };
            row[anchor] = 10.0 + jitter;
            row[anchor + 1] = jitter;
        }
        let rows: Vec<&[f32]> = raw.chunks_exact(D).collect();
        let affinities = Affinities::from_l2(&rows, 3, 5.0).unwrap();

        let mut same_cluster = 0usize;
        let mut total = 0usize;
        for i in 0..N {
            let (start, end) = (affinities.rows()[i], affinities.rows()[i + 1]);
            let cluster_i = if i < N / 2 { 0 } else { 1 };
            for &c in &affinities.columns()[start..end] {
                let cluster_c = if (c as usize) < N / 2 { 0 } else { 1 };
                if cluster_i == cluster_c {
                    same_cluster += 1;
                }
                total += 1;
            }
        }
        let purity = same_cluster as f32 / total as f32;
        assert!(
            purity > 0.8,
            "expected in-cluster neighbor purity above 0.8, got {purity} ({same_cluster}/{total})"
        );
    }

    /// Every `PcaError` variant reachable through `from_l2` fires on the matching bad input.
    /// `from_cosine` shares the same code path via `from_pca_metric`, so this doubles as
    /// coverage of the shared validation.
    #[test]
    fn from_l2_rejects_bad_input() {
        let empty: Vec<&[f32]> = Vec::new();
        assert!(matches!(
            Affinities::from_l2(&empty, 2, 5.0),
            Err(Err::NoRows)
        ));

        let ok = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ok_rows: Vec<&[f32]> = ok.chunks_exact(3).collect();
        assert!(matches!(
            Affinities::from_l2(&ok_rows, 0, 5.0),
            Err(Err::ZeroTargetDim)
        ));

        let ragged: Vec<&[f32]> = vec![&ok[0..3], &ok[0..2]];
        assert!(matches!(
            Affinities::from_l2(&ragged, 1, 5.0),
            Err(Err::RaggedRows { .. })
        ));

        // Perplexity too large: 3 * perplexity >= n after PCA (n = 2 rows).
        assert!(matches!(
            Affinities::from_l2(&ok_rows, 1, 5.0),
            Err(Err::PerplexityTooLarge { .. })
        ));
    }

    /// Two well-separated clusters of unit-norm rows in 40 dimensions, reduced to 3 via
    /// uncentered PCA. The pooled affinity graph must group each sample's top-k neighbors
    /// within its own cluster.
    #[test]
    fn from_cosine_recovers_cluster_structure() {
        const N: usize = 60;
        const D: usize = 40;
        let mut raw = vec![0.0f32; N * D];
        for i in 0..N {
            let cluster = if i < N / 2 { 0 } else { 1 };
            let angle = ((i as f32) * 0.31).sin() * 0.05;
            let row = &mut raw[i * D..(i + 1) * D];
            let anchor = if cluster == 0 { 0 } else { 20 };
            row[anchor] = 1.0 + angle;
            row[anchor + 1] = angle;
            // L2-normalize the row so it is a valid cosine input.
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            for v in row.iter_mut() {
                *v /= norm.max(1e-12);
            }
        }
        let rows: Vec<&[f32]> = raw.chunks_exact(D).collect();
        let affinities = Affinities::from_cosine(&rows, 3, 5.0).unwrap();

        let mut same_cluster = 0usize;
        let mut total = 0usize;
        for i in 0..N {
            let (start, end) = (affinities.rows()[i], affinities.rows()[i + 1]);
            let cluster_i = if i < N / 2 { 0 } else { 1 };
            for &c in &affinities.columns()[start..end] {
                let cluster_c = if (c as usize) < N / 2 { 0 } else { 1 };
                if cluster_i == cluster_c {
                    same_cluster += 1;
                }
                total += 1;
            }
        }
        let purity = same_cluster as f32 / total as f32;
        assert!(
            purity > 0.8,
            "expected in-cluster neighbor purity above 0.8, got {purity} ({same_cluster}/{total})"
        );
    }
}
