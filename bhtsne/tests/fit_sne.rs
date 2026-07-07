//! Integration tests for the FIt-SNE (interpolation) fit path.

use std::collections::HashSet;

use bhtsne::{Affinities, TsneBuilder};

mod common;
use common::{
    D, NO_DIMS, PERPLEXITY, bounding_box_diagonal, brute_force_neighbors, euclidean, lcg_samples,
    mean_point_distance,
};

#[test]
fn kl_divergence_after_fit_sne_is_finite_and_nonnegative() {
    const N: usize = 200;
    let data = lcg_samples(N, D, 11);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(250)
        .with_affinities(affinities)
        .fit_sne()
        .fit();

    let kl = fitted.kl_divergence();
    assert!(kl.is_finite() && kl >= 0.0, "kl divergence was {kl}");
}

/// End-to-end quality: FIt-SNE separates two trivially separable clusters. Same regression the
/// Barnes-Hut path is held to.
#[test]
fn fit_sne_separates_clusters() {
    const N_PER_CLUSTER: usize = 250;
    const DIM: usize = 10;

    let mut state = 1234_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
    };

    let mut data = Vec::with_capacity(2 * N_PER_CLUSTER * DIM);
    for cluster in 0..2 {
        let centre = if cluster == 0 { 0.0 } else { 30.0 };
        for _ in 0..N_PER_CLUSTER {
            for _ in 0..DIM {
                data.push(centre + 6.0 * next());
            }
        }
    }
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, 30.0, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .with_affinities(affinities)
        .fit_sne()
        .fit();
    let embedding = fitted.embedding();

    let n = 2 * N_PER_CLUSTER;
    let mut same_cluster = 0;
    for i in 0..n {
        let mut best = f32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = embedding[2 * i] - embedding[2 * j];
            let dy = embedding[2 * i + 1] - embedding[2 * j + 1];
            let d = dx * dx + dy * dy;
            if d < best {
                best = d;
                best_j = j;
            }
        }
        if (i < N_PER_CLUSTER) == (best_j < N_PER_CLUSTER) {
            same_cluster += 1;
        }
    }
    assert!(
        same_cluster as f64 / n as f64 > 0.95,
        "clusters not separated: only {same_cluster}/{n} points have a same-cluster nearest neighbor"
    );
}

/// FIt-SNE must not collapse the embedding onto a handful of points.
#[test]
fn fit_sne_does_not_collapse_embedding() {
    const N: usize = 400;
    const DIM: usize = 8;
    let data = lcg_samples(N, DIM, 23);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, 30.0, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .with_affinities(affinities)
        .fit_sne()
        .fit();
    let embedding = fitted.embedding();

    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "embedding contains non-finite values"
    );
    let distinct: HashSet<(i64, i64)> = embedding
        .chunks_exact(2)
        .map(|point| {
            (
                (point[0] * 100.0).round() as i64,
                (point[1] * 100.0).round() as i64,
            )
        })
        .collect();
    assert!(
        distinct.len() > N / 2,
        "embedding collapsed: only {} distinct positions for {N} points",
        distinct.len()
    );
}

/// FIt-SNE with `from_neighbors` reproduces the metric path within tolerance on a single-thread
/// pool, mirroring the Barnes-Hut test.
#[test]
fn fit_sne_with_neighbors_matches_vptree_path() {
    const N: usize = 120;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (reference, candidate) = pool.install(|| {
        let reference = {
            let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
            let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
                .epochs(100)
                .initial_embedding(&seed[..])
                .with_affinities(affinities)
                .fit_sne()
                .fit();
            fitted.embedding().to_vec()
        };
        let candidate = {
            let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
            let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
                .epochs(100)
                .initial_embedding(&seed[..])
                .with_affinities(affinities)
                .fit_sne()
                .fit();
            fitted.embedding().to_vec()
        };
        (reference, candidate)
    });

    let drift = mean_point_distance(&candidate, &reference, NO_DIMS as usize);
    let diagonal = bounding_box_diagonal(&reference, NO_DIMS as usize);
    assert!(
        drift <= 0.15 * diagonal + 1e-6,
        "from_neighbors fit_sne path diverged from metric path: drift {drift}, diagonal {diagonal}"
    );
}

/// An out-of-range neighbor index must be rejected up front by the FIt-SNE path.
#[test]
#[should_panic(expected = "out of range")]
fn fit_sne_with_neighbors_rejects_out_of_range_index() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, n_neighbors);
    neighbors[0][0].index = N;

    let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
    TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1)
        .with_affinities(affinities)
        .fit_sne()
        .fit();
}
