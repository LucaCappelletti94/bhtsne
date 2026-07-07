//! Integration tests for the Barnes-Hut fit path.

use std::collections::HashSet;

use bhtsne::{Affinities, TsneBuilder};

mod common;
use common::{
    D, EPOCHS, NO_DIMS, PERPLEXITY, THETA, bounding_box_diagonal, brute_force_neighbors, euclidean,
    lcg_samples, mean_point_distance,
};

#[test]
fn kl_divergence_after_barnes_hut_is_finite_and_nonnegative() {
    const N: usize = 60;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(100)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();

    let kl = fitted.kl_divergence();
    assert!(kl.is_finite() && kl >= 0.0, "{kl}");
}

/// Smoke test for the arena build and the force and reduction passes: the embedding stays finite
/// and correctly sized after a short Barnes-Hut fit.
#[test]
fn parallel_barnes_hut_build_smoke() {
    const N: usize = 160;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 5);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(3)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();

    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * NO_DIMS as usize);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

/// Regression: `epoch_callback` is invoked once per epoch, in order, with the current embedding
/// buffer. The final snapshot must match the value the fit returns.
#[test]
fn epoch_callback_reports_each_barnes_hut_epoch() {
    const N: usize = 60;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 100;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let mut epochs_seen: Vec<usize> = Vec::new();
    let mut last_snapshot: Vec<f32> = Vec::new();

    let embedding = {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(RUN_EPOCHS)
            .epoch_callback(|epoch, snapshot| {
                assert_eq!(snapshot.len(), N * NO_DIMS as usize);
                epochs_seen.push(epoch);
                last_snapshot.clear();
                last_snapshot.extend_from_slice(snapshot);
            })
            .with_affinities(affinities)
            .bhtsne(THETA)
            .fit();
        fitted.embedding().to_vec()
    };

    assert_eq!(epochs_seen, (0..RUN_EPOCHS).collect::<Vec<usize>>());
    assert_eq!(last_snapshot, embedding);
}

/// The warm-start builder shortcut resumes from the supplied initial embedding rather than a
/// random one, so the first epoch stays close to the seed and the run does not restart from
/// scratch.
#[test]
fn warm_start_begins_from_initial_embedding_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let seed = {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(300)
            .with_affinities(affinities)
            .bhtsne(THETA)
            .fit();
        fitted.embedding().to_vec()
    };

    let mut first_snapshot: Vec<f32> = Vec::new();
    {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
        TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(5)
            .stop_lying_epoch(0)
            .momentum_switch_epoch(0)
            .initial_embedding(&seed[..])
            .epoch_callback(|epoch, snapshot| {
                if epoch == 0 {
                    first_snapshot.extend_from_slice(snapshot);
                }
            })
            .with_affinities(affinities)
            .bhtsne(THETA)
            .fit();
    }

    let dim = NO_DIMS as usize;
    let displacement = mean_point_distance(&first_snapshot, &seed, dim);
    let diagonal = bounding_box_diagonal(&seed, dim);
    assert!(
        displacement < 0.05 * diagonal,
        "first epoch strayed {displacement} from the seed, its bounding box diagonal is {diagonal}"
    );

    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// The initial embedding length must match `n_samples * NO_DIMS`; a shorter buffer is rejected
/// up front.
#[test]
#[should_panic(expected = "initial embedding has")]
fn warm_start_rejects_wrong_length_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1)
        .initial_embedding([0.0; 7])
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
}

/// A `stop_lying_epoch` of zero disables the early exaggeration factor: the first-epoch step is
/// small compared to the exaggerated version.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let seed = {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(300)
            .with_affinities(affinities)
            .bhtsne(THETA)
            .fit();
        fitted.embedding().to_vec()
    };

    let first_step = |stop_lying_epoch: usize| -> f32 {
        let mut first_snapshot: Vec<f32> = Vec::new();
        {
            let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });
            TsneBuilder::<f32, &[f32]>::new(&samples)
                .epochs(1)
                .stop_lying_epoch(stop_lying_epoch)
                .initial_embedding(&seed[..])
                .epoch_callback(|_epoch, snapshot| {
                    first_snapshot.extend_from_slice(snapshot);
                })
                .with_affinities(affinities)
                .bhtsne(THETA)
                .fit();
        }
        mean_point_distance(&first_snapshot, &seed, NO_DIMS as usize)
    };

    let exaggerated = first_step(1000);
    let truthful = first_step(0);
    assert!(
        truthful < exaggerated / 3.0,
        "first epoch still exaggerated: moved {truthful} against {exaggerated} with 12x P values"
    );
}

/// Feeding the exact neighbors the vp-tree would find reproduces the metric fit within tolerance.
/// Column ordering differs between `symmetrize_sparse_matrix` (dense path) and `symmetrize_csr`
/// (ragged path), so the bit-comparison is done on a single-thread pool and the two embeddings
/// are compared with drift tolerance.
#[test]
fn barnes_hut_with_neighbors_matches_vptree_path() {
    const N: usize = 80;
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
                .bhtsne(THETA)
                .fit();
            fitted.embedding().to_vec()
        };
        let candidate = {
            let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
            let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
                .epochs(100)
                .initial_embedding(&seed[..])
                .with_affinities(affinities)
                .bhtsne(THETA)
                .fit();
            fitted.embedding().to_vec()
        };
        (reference, candidate)
    });

    let drift = mean_point_distance(&candidate, &reference, NO_DIMS as usize);
    let diagonal = bounding_box_diagonal(&reference, NO_DIMS as usize);
    assert!(
        drift <= 0.15 * diagonal + 1e-6,
        "from_neighbors path diverged from metric path: drift {drift}, diagonal {diagonal}"
    );
}

/// An out-of-range neighbor index must be rejected up front by the affinity constructor.
#[test]
#[should_panic(expected = "out of range")]
fn barnes_hut_with_neighbors_rejects_out_of_range_index() {
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
        .bhtsne(THETA)
        .fit();
}

/// Two well-separated clusters in 10 dimensions must remain separated in the 2D embedding: at
/// least 95% of points must have their nearest embedded neighbor come from the same cluster.
#[test]
fn barnes_hut_separates_clusters_at_large_input_scale() {
    const N_PER_CLUSTER: usize = 150;
    const DIM: usize = 10;

    let mut state = 42_u64;
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

    let affinities = Affinities::from_metric(&samples, 30.0, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .with_affinities(affinities)
        .bhtsne(THETA)
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

/// With supplied neighbors (removing vp-tree randomness) and a fixed seed, two Barnes-Hut runs
/// must agree within a small fraction of the embedding scale. Determinism is relaxed for the
/// arena and reductions, so this is a tolerance rather than bit comparison.
#[test]
fn barnes_hut_is_stable_run_to_run() {
    const N: usize = 600;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let run = || {
        let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(150)
            .initial_embedding(&seed[..])
            .with_affinities(affinities)
            .bhtsne(THETA)
            .fit();
        fitted.embedding().to_vec()
    };

    let first = run();
    let second = run();
    let drift = mean_point_distance(&first, &second, NO_DIMS as usize);
    let diagonal = bounding_box_diagonal(&first, NO_DIMS as usize);
    assert!(
        drift <= 0.05 * diagonal + 1e-4,
        "two runs diverged: mean drift {drift} exceeds tolerance for diagonal {diagonal}"
    );
}

/// Regression: the parallel arena build must not lose or invent points. Point conservation is
/// checked directly through the arena's root count. The offset cloud isolates the failure mode.
#[test]
fn arena_build_maintains_invariants() {
    const N: usize = 2_000;
    let mut data = lcg_samples(N, 2, 17);
    for value in data.iter_mut() {
        *value += 100.0;
    }

    let arena = barnes_hut_tree::BarnesHutTree::<f32, u64, 2>::new_uniform(&data);
    assert_eq!(arena.root_count(), N, "arena lost or invented points");
}

/// Regression: corrupted repulsive forces would let attraction collapse the whole embedding onto
/// a handful of coordinates. A healthy run keeps more than half the points at distinct positions.
#[test]
fn barnes_hut_does_not_collapse_embedding() {
    const N: usize = 500;
    const DIM: usize = 8;
    let data = lcg_samples(N, DIM, 23);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, 30.0, |a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1000)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();

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

/// Round trip: run bhtsne, extract affinities, inject them plus the resulting embedding into a
/// second builder. The continuation stays closer to the first embedding than a random-init run.
#[test]
fn affinities_round_trip_barnes_hut() {
    const N: usize = 200;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let affinities1 = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted1 = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities1)
        .bhtsne(THETA)
        .fit();
    let embedding1 = fitted1.embedding().to_vec();
    let affinities = fitted1.affinities().clone();

    let fitted2 = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .initial_embedding(embedding1.clone())
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding2 = fitted2.embedding().to_vec();

    let affinities_rand = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_rand = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .with_affinities(affinities_rand)
        .bhtsne(THETA)
        .fit();
    let embedding_rand = fitted_rand.embedding().to_vec();

    let dist_warm = mean_point_distance(&embedding1, &embedding2, D);
    let dist_rand = mean_point_distance(&embedding1, &embedding_rand, D);
    assert!(
        dist_warm < dist_rand,
        "warm start ({dist_warm}) should be closer to seed than random ({dist_rand})"
    );

    let data10 = &data[..D * 10];
    let dist1 = mean_point_distance(data10, &embedding1[..D * 10], D);
    let dist2 = mean_point_distance(data10, &embedding2[..D * 10], D);
    assert!(
        (dist1 - dist2).abs() < dist1 * 0.5,
        "cluster structure changed too much: {dist1} vs {dist2}"
    );
}

/// A second bhtsne call with cloned affinities from a reference run produces the same embedding
/// as a plain bhtsne call within tolerance for parallel-reduction noise.
#[test]
fn affinities_equivalence_first_step() {
    const N: usize = 100;
    const CAPTURE: usize = 1;

    let data = lcg_samples(N, D, 99);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let affinities_ref = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_ref = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities_ref)
        .bhtsne(THETA)
        .fit();
    let affinities = fitted_ref.affinities().clone();
    let seed = fitted_ref.embedding().to_vec();

    let affinities_a = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_a = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed.clone())
        .with_affinities(affinities_a)
        .bhtsne(THETA)
        .fit();
    let result_a = fitted_a.embedding().to_vec();

    let fitted_b = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let result_b = fitted_b.embedding().to_vec();

    let max_diff: f32 = result_a
        .iter()
        .zip(result_b.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let scale = result_a
        .iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        max_diff / scale < 1e-3,
        "precomputed path diverged from reference: max_diff={max_diff}, scale={scale}"
    );
}

/// `affinities()` returns values summing to about one regardless of run length. The stored graph
/// is the pristine P matrix, so a short and a long run must expose the same values.
#[test]
fn affinities_pristine_independent_of_run_length() {
    const N: usize = 100;

    let data = lcg_samples(N, D, 7);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let affinities_short = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_short = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(50)
        .with_affinities(affinities_short)
        .bhtsne(THETA)
        .fit();
    let aff_short = fitted_short.affinities();
    let sum_short: f32 = aff_short.values().iter().sum();

    let affinities_long = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_long = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1000)
        .with_affinities(affinities_long)
        .bhtsne(THETA)
        .fit();
    let aff_long = fitted_long.affinities();
    let sum_long: f32 = aff_long.values().iter().sum();

    assert!(
        (sum_short - 1.0).abs() < 0.05,
        "short run affinities sum to {sum_short}, expected ~1"
    );
    assert!(
        (sum_long - 1.0).abs() < 0.05,
        "long run affinities sum to {sum_long}, expected ~1"
    );
    assert_eq!(
        aff_short.rows(),
        aff_long.rows(),
        "row structure differs between short and long runs"
    );
    assert_eq!(
        aff_short.columns(),
        aff_long.columns(),
        "column structure differs between short and long runs"
    );
    for (a, b) in aff_short.values().iter().zip(aff_long.values().iter()) {
        assert!((a - b).abs() < 1e-6, "value differs: {a} vs {b}");
    }
}

/// Smoke: Barnes-Hut fit in 3 output dimensions.
#[test]
fn barnes_hut_runs_in_three_dimensions() {
    const N: usize = 200;
    const DIN: usize = 5;
    let data = lcg_samples(N, DIN, 43);
    let samples: Vec<&[f32]> = data.chunks(DIN).collect();
    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32], 3>::new(&samples)
        .epochs(50)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 3);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

#[test]
fn barnes_hut_runs_in_four_dimensions() {
    const N: usize = 200;
    const DIN: usize = 5;
    let data = lcg_samples(N, DIN, 42);
    let samples: Vec<&[f32]> = data.chunks(DIN).collect();
    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32], 4>::new(&samples)
        .epochs(50)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 4);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

#[test]
fn barnes_hut_runs_in_five_dimensions() {
    const N: usize = 200;
    const DIN: usize = 6;
    let data = lcg_samples(N, DIN, 44);
    let samples: Vec<&[f32]> = data.chunks(DIN).collect();
    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32], 5>::new(&samples)
        .epochs(50)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 5);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

#[test]
fn barnes_hut_runs_in_six_dimensions() {
    const N: usize = 200;
    const DIN: usize = 7;
    let data = lcg_samples(N, DIN, 45);
    let samples: Vec<&[f32]> = data.chunks(DIN).collect();
    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32], 6>::new(&samples)
        .epochs(50)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 6);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

#[test]
fn barnes_hut_runs_in_seven_dimensions() {
    const N: usize = 200;
    const DIN: usize = 8;
    let data = lcg_samples(N, DIN, 46);
    let samples: Vec<&[f32]> = data.chunks(DIN).collect();
    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32], 7>::new(&samples)
        .epochs(50)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 7);
    assert!(embedding.iter().all(|v| v.is_finite()));
}
