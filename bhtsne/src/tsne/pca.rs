//! Randomized PCA reduction via subspace iteration, used internally by the
//! [`Affinities::from_l2`] and [`Affinities::from_cosine`] sugar constructors.
//!
//! Reuses [`splitmix_unit`], [`orthonormalize_block`], and [`jacobi_eigen`] from the spectral
//! solver. Randomized subspace iteration on the covariance operator (with optional centering)
//! plus Rayleigh-Ritz on the final block; converges in a handful of rounds for the
//! class-structured feature spaces we care about.
//!
//! [`Affinities::from_l2`]: crate::Affinities::from_l2
//! [`Affinities::from_cosine`]: crate::Affinities::from_cosine

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::Float;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::error::PcaError;
use crate::tsne::spectral::{jacobi_eigen, orthonormalize_block, splitmix_unit};

/// Subspace iterations. Empirically enough to converge for typical class-structured feature
/// matrices; the leading covariance eigenvalues separate well after a handful of rounds.
const DEFAULT_ROUNDS: usize = 8;

/// Extra columns carried beyond `target_dim` for numerical stability of the subspace iteration
/// when the leading eigenvalues are tightly clustered. Absorbs the unwanted eigenvectors the
/// iteration amplifies alongside the wanted ones.
const OVERSAMPLE: usize = 4;

/// Deterministic seed for the random Gaussian initialization. Fixed so runs are reproducible.
const SEED: u64 = 0x00C0_FFEE_1234_5678;

/// Reduces `rows` to `target_dim` dimensions via randomized subspace iteration. If `center` is
/// true the column means are subtracted first (standard PCA); otherwise the iteration runs on
/// the raw Gram operator `X^T X` (equivalent to truncated SVD of X, appropriate for cosine
/// after L2 normalization since centering would break the unit-norm invariant).
///
/// Output is row-major `n * min(target_dim, d)`. If `target_dim >= d`, the reduction is a
/// passthrough: no eigendecomposition is performed and the (optionally centered) input is
/// returned as-is. Callers that hard-coded `target_dim` must therefore chunk by
/// `target_dim.min(d)` when interpreting the output.
pub(crate) fn pca_reduce<T>(
    rows: &[&[T]],
    target_dim: usize,
    center: bool,
) -> Result<Vec<T>, PcaError>
where
    T: Float + Default + Send + Sync + Sum + AddAssign + SubAssign + MulAssign + DivAssign + Copy,
{
    if rows.is_empty() {
        return Err(PcaError::NoRows);
    }
    if target_dim == 0 {
        return Err(PcaError::ZeroTargetDim);
    }
    let n = rows.len();
    let d = rows[0].len();
    for (i, r) in rows.iter().enumerate() {
        if r.len() != d {
            return Err(PcaError::RaggedRows {
                row: i,
                expected: d,
                actual: r.len(),
            });
        }
    }

    // Flatten X into a row-major buffer we own; needed for centering and repeat matvecs.
    // The row copies run in parallel so the page-fault cost of the fresh `n * d` allocation
    // is spread across cores rather than serialized on the writing thread.
    let mut x: Vec<T> = vec![T::zero(); n * d];
    x.par_chunks_mut(d)
        .zip(rows.par_iter())
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    if center {
        subtract_column_means(&mut x, n, d);
    }

    // Passthrough: PCA cannot produce more than `d` distinct components. Asking for
    // `target_dim >= d` returns the (optionally centered) input directly.
    if target_dim >= d {
        return Ok(x);
    }

    let k = target_dim;
    // Cap the working block width at d: can't have more subspace columns than features.
    let kk = (k + OVERSAMPLE).min(d);

    // Random Gaussian initial Q (d * kk row-major), then orthonormalize. `v0 = 0` gives the
    // spectral orthonormalizer's deflation term a no-op contribution, reducing it to plain
    // classical Gram-Schmidt (twice, for stability).
    let mut q: Vec<T> = (0..d * kk)
        .map(|i| splitmix_unit::<T>(SEED.wrapping_add(i as u64)))
        .collect();
    let v0 = vec![T::zero(); d];
    orthonormalize_block(&mut q, d, kk, &v0, SEED.wrapping_add(0x1000_0000));

    // Subspace iteration: repeatedly apply X^T X to the block, orthonormalize.
    for round in 0..DEFAULT_ROUNDS {
        let y = x_times(&x, &q, n, d, kk);
        let mut q_new = x_transpose_times(&x, &y, n, d, kk);
        orthonormalize_block(
            &mut q_new,
            d,
            kk,
            &v0,
            SEED.wrapping_add(0x2000_0000 + round as u64),
        );
        q = q_new;
    }

    // Rayleigh-Ritz on the reduced basis: build `B = (XQ)^T (XQ)`, eigendecompose, and take the
    // leading `k` eigenvectors to sort by descending eigenvalue.
    let y = x_times(&x, &q, n, d, kk);
    let mut b = gram(&y, n, kk);
    let e = jacobi_eigen(&mut b, kk);
    // After jacobi_eigen, `b` holds eigenvalues on the diagonal and near-zero off-diagonals.
    let mut order: Vec<usize> = (0..kk).collect();
    order.sort_by(|&a, &c| {
        b[c * kk + c]
            .partial_cmp(&b[a * kk + a])
            .expect("non-NaN eigenvalue")
    });
    let top: Vec<usize> = order.into_iter().take(k).collect();

    // Build a (kk * k) row-major matrix of the selected eigenvectors (which live as columns of
    // `e`), then project `y` through it: `output = y @ top_e` gives (n * k) row-major.
    let mut top_e = vec![T::zero(); kk * k];
    for row in 0..kk {
        for (idx, &col) in top.iter().enumerate() {
            top_e[row * k + idx] = e[row * kk + col];
        }
    }
    Ok(matmul(&y, &top_e, n, kk, k))
}

/// In-place column-mean subtraction over a row-major `n * d` matrix.
fn subtract_column_means<T>(x: &mut [T], n: usize, d: usize)
where
    T: Float + AddAssign + SubAssign + DivAssign + Send + Sync,
{
    let n_t = T::from(n).expect("row count fits in T");
    let means: Vec<T> = (0..d)
        .into_par_iter()
        .map(|col| {
            let mut sum = T::zero();
            for i in 0..n {
                sum += x[i * d + col];
            }
            sum / n_t
        })
        .collect();
    x.par_chunks_mut(d).for_each(|row| {
        for col in 0..d {
            row[col] -= means[col];
        }
    });
}

/// `Y = X @ Q` where X is n * d and Q is d * kk, both row-major. Rows are computed in parallel.
fn x_times<T>(x: &[T], q: &[T], n: usize, d: usize, kk: usize) -> Vec<T>
where
    T: Float + Default + AddAssign + Send + Sync + Copy,
{
    let mut y = vec![T::zero(); n * kk];
    y.par_chunks_mut(kk).enumerate().for_each(|(i, y_row)| {
        let x_row = &x[i * d..(i + 1) * d];
        for f in 0..d {
            let xif = x_row[f];
            let q_row = &q[f * kk..(f + 1) * kk];
            for j in 0..kk {
                y_row[j] += xif * q_row[j];
            }
        }
    });
    y
}

/// `Q' = X^T @ Y` where X is n * d and Y is n * kk, both row-major, output d * kk row-major.
/// Rows of the output are computed in parallel; each output row iterates over all samples so
/// no cross-thread reduction is needed.
fn x_transpose_times<T>(x: &[T], y: &[T], n: usize, d: usize, kk: usize) -> Vec<T>
where
    T: Float + Default + AddAssign + Send + Sync + Copy,
{
    let mut out = vec![T::zero(); d * kk];
    out.par_chunks_mut(kk).enumerate().for_each(|(f, out_row)| {
        for i in 0..n {
            let xif = x[i * d + f];
            let y_row = &y[i * kk..(i + 1) * kk];
            for j in 0..kk {
                out_row[j] += xif * y_row[j];
            }
        }
    });
    out
}

/// `B = Y^T @ Y`, a symmetric `kk * kk` row-major Gram matrix. Deterministic: rows of B are
/// filled independently and each is a sequential sum over samples.
fn gram<T>(y: &[T], n: usize, kk: usize) -> Vec<T>
where
    T: Float + Default + AddAssign + Send + Sync + Copy,
{
    let mut b = vec![T::zero(); kk * kk];
    b.par_chunks_mut(kk).enumerate().for_each(|(i, b_row)| {
        for s in 0..n {
            let ysi = y[s * kk + i];
            let ys_row = &y[s * kk..(s + 1) * kk];
            for j in 0..kk {
                b_row[j] += ysi * ys_row[j];
            }
        }
    });
    b
}

/// `C = A @ B` where A is n * k1 and B is k1 * k2, both row-major, output n * k2.
fn matmul<T>(a: &[T], b: &[T], n: usize, k1: usize, k2: usize) -> Vec<T>
where
    T: Float + Default + AddAssign + Send + Sync + Copy,
{
    let mut c = vec![T::zero(); n * k2];
    c.par_chunks_mut(k2).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k1..(i + 1) * k1];
        for f in 0..k1 {
            let aif = a_row[f];
            let b_row = &b[f * k2..(f + 1) * k2];
            for j in 0..k2 {
                c_row[j] += aif * b_row[j];
            }
        }
    });
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic two-cluster synthetic in 30 dims: rows 0..30 sit near `+e_0` (unit vector on
    /// axis 0) plus tiny noise on the other axes, rows 30..60 sit near `-e_0`. The leading
    /// principal component must recover the axis-0 direction and give the two clusters opposite
    /// signs on the first output column.
    #[test]
    fn top_pc_separates_two_clusters() {
        let n = 60;
        let d = 30;
        let mut raw: Vec<f32> = Vec::with_capacity(n * d);
        for i in 0..n {
            let sign = if i < 30 { 1.0 } else { -1.0 };
            for j in 0..d {
                let mut val = if j == 0 { sign } else { 0.0 };
                let noise: f32 = splitmix_unit(0x00C0_FFEE_u64.wrapping_add((i * d + j) as u64));
                val += 0.01 * noise;
                raw.push(val);
            }
        }
        let rows: Vec<&[f32]> = raw.chunks_exact(d).collect();
        let reduced = pca_reduce(&rows, 2, true).expect("pca");
        assert_eq!(reduced.len(), n * 2);

        let mut pos = 0;
        let mut neg = 0;
        for (i, chunk) in reduced.chunks_exact(2).enumerate() {
            if i < 30 {
                if chunk[0] > 0.0 {
                    pos += 1;
                } else {
                    neg += 1;
                }
            }
        }
        assert!(
            pos > 25 || neg > 25,
            "cluster 0 should sit almost entirely on one sign of PC1 (pos={pos}, neg={neg})"
        );
    }

    #[test]
    fn errors_on_bad_input() {
        let empty: Vec<&[f32]> = Vec::new();
        assert!(matches!(pca_reduce(&empty, 2, true), Err(PcaError::NoRows)));

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let rows_ok: Vec<&[f32]> = a.chunks_exact(2).collect();
        assert!(matches!(
            pca_reduce(&rows_ok, 0, true),
            Err(PcaError::ZeroTargetDim)
        ));
        // target_dim >= d is a passthrough, not an error.
        let pass = pca_reduce(&rows_ok, 5, false).unwrap();
        assert_eq!(
            pass, a,
            "passthrough must return the input verbatim (uncentered)"
        );

        let ragged: Vec<f32> = vec![1.0, 2.0, 3.0];
        let ragged_rows: Vec<&[f32]> = vec![&ragged[0..2], &ragged[0..3]];
        assert!(matches!(
            pca_reduce(&ragged_rows, 1, true),
            Err(PcaError::RaggedRows { .. })
        ));
    }

    /// Runs on the same input twice; the output must match exactly (bit-exact determinism).
    #[test]
    fn output_is_deterministic() {
        let n = 40;
        let d = 12;
        let raw: Vec<f32> = (0..n * d)
            .map(|i| splitmix_unit(0xDEAD_BEEF_u64.wrapping_add(i as u64)))
            .collect();
        let rows: Vec<&[f32]> = raw.chunks_exact(d).collect();
        let a = pca_reduce(&rows, 3, true).unwrap();
        let b = pca_reduce(&rows, 3, true).unwrap();
        assert_eq!(a, b);
    }

    /// Rank-`r` data embedded in high-d must have its pairwise Euclidean distances exactly
    /// preserved (up to numerical error) when PCA-reduced back to `r` dimensions. Failure here
    /// means the reduction is not producing an orthonormal basis of the true eigenspace.
    #[test]
    fn preserves_pairwise_distances_on_rank_r_data() {
        let n = 60;
        let r = 3;
        let d = 40;
        // X = A @ B where A is n*r and B is r*d — exactly rank r.
        let a: Vec<f32> = (0..n * r)
            .map(|i| splitmix_unit(0xA000_u64.wrapping_add(i as u64)))
            .collect();
        let b: Vec<f32> = (0..r * d)
            .map(|i| splitmix_unit(0xB000_u64.wrapping_add(i as u64)))
            .collect();
        let mut x = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += a[i * r + k] * b[k * d + j];
                }
                x[i * d + j] = sum;
            }
        }
        let rows: Vec<&[f32]> = x.chunks_exact(d).collect();

        let reduced = pca_reduce(&rows, r, true).unwrap();
        assert_eq!(reduced.len(), n * r);

        // Ground-truth pairwise distances on centered X.
        let mut col_means = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d {
                col_means[j] += x[i * d + j];
            }
        }
        for m in col_means.iter_mut() {
            *m /= n as f32;
        }
        let xc: Vec<f32> = (0..n * d).map(|k| x[k] - col_means[k % d]).collect();

        let mut max_rel = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut orig = 0.0f32;
                for k in 0..d {
                    let diff = xc[i * d + k] - xc[j * d + k];
                    orig += diff * diff;
                }
                let orig = orig.sqrt();

                let mut reduced_d = 0.0f32;
                for k in 0..r {
                    let diff = reduced[i * r + k] - reduced[j * r + k];
                    reduced_d += diff * diff;
                }
                let reduced_d = reduced_d.sqrt();

                let rel = (orig - reduced_d).abs() / (orig + 1e-6);
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        // On rank-r data reduced to r dims, distances should be preserved to within numerical
        // error of the subspace iteration convergence. Bar generously: 1% is well above the
        // measured f32 baseline (~1e-5) if the algorithm is correct.
        assert!(
            max_rel < 0.01,
            "max relative pairwise distance error {max_rel} is too large; PCA is not recovering the rank-{r} eigenspace"
        );
    }
}
