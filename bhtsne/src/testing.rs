//! Shared `#[cfg(test)]` fixtures for the inline unit tests inside `src/`.
//!
//! Integration tests under `tests/` have their own copy under `tests/common/mod.rs` because
//! `#[cfg(test)]` items in the library crate are only visible to the library's own unit tests,
//! never to the separate `tests/*.rs` binaries.

use crate::Neighbor;

/// Deterministic pseudo random samples. LCG seeded on `state` so tests are reproducible.
pub(crate) fn lcg_samples(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((state >> 33) as f32 / u32::MAX as f32) - 0.5);
    }
    data
}

/// Euclidean distance between two f32 samples.
pub(crate) fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Exact `n_neighbors` nearest neighbors per sample, sorted ascending by distance and excluding
/// the sample itself.
pub(crate) fn brute_force_neighbors(
    samples: &[&[f32]],
    n_neighbors: usize,
) -> Vec<Vec<Neighbor<f32>>> {
    (0..samples.len())
        .map(|i| {
            let mut distances: Vec<(usize, f32)> = (0..samples.len())
                .filter(|&j| j != i)
                .map(|j| (j, euclidean(samples[i], samples[j])))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(n_neighbors);
            distances
                .into_iter()
                .map(|(index, distance)| Neighbor { index, distance })
                .collect()
        })
        .collect()
}
