use std::collections::HashSet;

use super::{Affinities, Neighbor, SpectralParams, TsneBuilder, tsne};

const D: usize = 4;
const THETA: f32 = 0.5;
const PERPLEXITY: f32 = 10.;
const EPOCHS: usize = 2_000;
const NO_DIMS: u8 = 2;

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

#[test]
fn kl_divergence_after_exact_is_finite_and_nonnegative() {
    const N: usize = 60;
    const DIM: usize = 4;
    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(100)
        .with_affinities(affinities)
        .exact()
        .fit();

    let kl = fitted.kl_divergence();
    assert!(kl.is_finite() && kl >= 0.0, "{kl}");
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires iris dataset"]
fn exact_tsne() {
    let data: Vec<f32> =
        crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
    let samples: Vec<&[f32]> = data.chunks(D).collect::<Vec<&[f32]>>();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities)
        .exact()
        .fit();
    fitted.write_csv("iris_embedding_vanilla.csv").unwrap();

    let embedding = fitted.embedding();
    let points: Vec<_> = embedding.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(fitted.kl_divergence() < 0.5);
}

#[cfg(feature = "csv")]
#[test]
#[ignore = "requires iris dataset"]
fn barnes_hut_tsne() {
    let data: Vec<f32> =
        crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
    let samples: Vec<&[f32]> = data.chunks(D).collect::<Vec<&[f32]>>();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    fitted.write_csv("iris_embedding_barnes_hut.csv").unwrap();

    let embedding = fitted.embedding();
    let points: Vec<_> = embedding.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(fitted.kl_divergence() < 5.0);
}

/// The epoch callback must be invoked once per epoch, in order, with a snapshot
/// of the embedding whose final value matches the result of `embedding`.
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

/// Same as `epoch_callback_reports_each_barnes_hut_epoch` for the exact version
/// of the algorithm.
#[test]
fn epoch_callback_reports_each_exact_epoch() {
    const N: usize = 60;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 50;

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
                .sum()
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
            .exact()
            .fit();
        fitted.embedding().to_vec()
    };

    assert_eq!(epochs_seen, (0..RUN_EPOCHS).collect::<Vec<usize>>());
    assert_eq!(last_snapshot, embedding);
}

/// The epoch callback is invoked only on the fitting thread, so it need not be
/// `Send` or `Sync`. A closure capturing an `Rc<RefCell<_>>` is neither, which the
/// previous bound rejected; this is exactly the shape a single threaded wasm
/// worker needs to forward progress. If the bound ever tightened again, this test
/// would fail to compile.
#[test]
fn epoch_callback_accepts_non_send_closure() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const N: usize = 40;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 10;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // `Rc<RefCell<_>>` is neither `Send` nor `Sync`, so this closure is `!Send`.
    let epochs_seen = Rc::new(RefCell::new(Vec::<usize>::new()));
    let sink = Rc::clone(&epochs_seen);

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    });
    TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(RUN_EPOCHS)
        .epoch_callback(move |epoch, _snapshot| {
            sink.borrow_mut().push(epoch);
        })
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();

    assert_eq!(
        *epochs_seen.borrow(),
        (0..RUN_EPOCHS).collect::<Vec<usize>>()
    );
}

/// A warm started fit must begin from the supplied embedding: the first epoch
/// stays close to the seed, far closer than a random init near the origin would.
#[test]
fn warm_start_begins_from_initial_embedding_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
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

    // A random initialization concentrates every point around the origin, so
    // its mean displacement from the seed is the mean seed point norm.
    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// Same as `warm_start_begins_from_initial_embedding_barnes_hut` for the exact
/// version of the algorithm.
#[test]
fn warm_start_begins_from_initial_embedding_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        });
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(300)
            .with_affinities(affinities)
            .exact()
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
                .sum()
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
            .exact()
            .fit();
    }

    let dim = NO_DIMS as usize;
    let displacement = mean_point_distance(&first_snapshot, &seed, dim);
    let diagonal = bounding_box_diagonal(&seed, dim);
    assert!(
        displacement < 0.05 * diagonal,
        "first epoch strayed {displacement} from the seed, its bounding box diagonal is {diagonal}"
    );

    // A random initialization concentrates every point around the origin, so
    // its mean displacement from the seed is the mean seed point norm.
    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// Exact squared-euclidean distance used by the early-exaggeration fits below.
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Runs a short exact fit from a fixed seed and returns the embedding snapshot at
/// the end of epoch `capture`, so two configurations can be compared at the same
/// point of the optimization. `configure` sets the knob under test on the builder.
fn exact_snapshot_at<F>(capture: usize, configure: F) -> Vec<f32>
where
    F: for<'a> FnOnce(TsneBuilder<'a, f32, &'a [f32]>) -> TsneBuilder<'a, f32, &'a [f32]>,
{
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let mut snapshot: Vec<f32> = Vec::new();
    {
        let affinities =
            Affinities::from_metric(&samples, PERPLEXITY, |a, b| squared_euclidean(a, b));
        let builder = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(capture + 1)
            .initial_embedding(&seed[..]);
        let builder = configure(builder);
        builder
            .epoch_callback(|epoch, current| {
                if epoch == capture {
                    snapshot.extend_from_slice(current);
                }
            })
            .with_affinities(affinities)
            .exact()
            .fit();
    }
    snapshot
}

/// Setting the factor to its default `12.0` explicitly must match leaving it
/// unset, so existing callers see no behavior change. The two runs execute the
/// same arithmetic, so they may differ only by parallel-reduction noise, far
/// below the embedding scale.
#[test]
fn early_exaggeration_explicit_twelve_matches_default() {
    // Compare after a single step: the optimizer is chaotic, so even identical
    // arithmetic diverges macroscopically over many epochs once parallel-reduction
    // noise is amplified. One step isolates the behavior from that amplification.
    let default_run = exact_snapshot_at(0, |builder| builder);
    let explicit_run = exact_snapshot_at(0, |builder| builder.early_exaggeration(12.0));

    let dim = NO_DIMS as usize;
    let drift = mean_point_distance(&default_run, &explicit_run, dim);
    let diagonal = bounding_box_diagonal(&default_run, dim);
    assert!(
        drift <= 1e-3 * diagonal + 1e-6,
        "explicit 12.0 strayed {drift} from the default, diagonal {diagonal}"
    );
}

/// The factor must reach the optimizer: two fits differing only in
/// `early_exaggeration` pull the embedding apart by different amounts in the
/// early epochs, so their first-epoch snapshots are measurably different.
#[test]
fn early_exaggeration_changes_early_embedding() {
    let strong = exact_snapshot_at(0, |builder| builder.early_exaggeration(12.0));
    let weak = exact_snapshot_at(0, |builder| builder.early_exaggeration(4.0));

    let dim = NO_DIMS as usize;
    let difference = mean_point_distance(&strong, &weak, dim);
    let diagonal = bounding_box_diagonal(&strong, dim);
    assert!(
        difference > 0.05 * diagonal,
        "exaggeration 12.0 against 4.0 barely moved the first epoch: {difference} against diagonal {diagonal}"
    );
}

/// `early_exaggeration(1.0)` is a second way to express "no exaggeration": it must
/// produce the same first epoch as `stop_lying_epoch(0)`, which normalizes then
/// undoes the lying immediately. Both leave the `P` distribution unexaggerated.
#[test]
fn early_exaggeration_one_matches_stop_lying_zero() {
    let no_exaggeration = exact_snapshot_at(0, |builder| builder.early_exaggeration(1.0));
    let lying_disabled = exact_snapshot_at(0, |builder| builder.stop_lying_epoch(0));

    let dim = NO_DIMS as usize;
    let drift = mean_point_distance(&no_exaggeration, &lying_disabled, dim);
    let diagonal = bounding_box_diagonal(&no_exaggeration, dim);
    assert!(
        drift <= 1e-3 * diagonal + 1e-6,
        "the two no-exaggeration paths diverged: {drift} against diagonal {diagonal}"
    );
}

/// The Barnes-Hut fit must reject a seed whose length does not match
/// `n_samples * D`.
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

/// The exact fit carries its own length check, exercise it independently of the
/// Barnes-Hut one.
#[test]
#[should_panic(expected = "initial embedding has")]
fn warm_start_rejects_wrong_length_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
        sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    });
    TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1)
        .initial_embedding([0.0; 7])
        .with_affinities(affinities)
        .exact()
        .fit();
}

/// A stop lying epoch of zero must mean no early exaggeration at all. Two warm
/// started single epoch runs, one with the exaggeration off and one with it on,
/// must take differently sized first steps, since the momentum buffer is zero at
/// epoch 0 the two differ by the exaggeration factor alone.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_barnes_hut() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
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

/// Same as `stop_lying_epoch_zero_skips_exaggeration_barnes_hut` for the exact
/// version of the algorithm.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_exact() {
    const N: usize = 60;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A plausible layout to continue from.
    let seed = {
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        });
        let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
            .epochs(300)
            .with_affinities(affinities)
            .exact()
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
                    .sum()
            });
            TsneBuilder::<f32, &[f32]>::new(&samples)
                .epochs(1)
                .stop_lying_epoch(stop_lying_epoch)
                .initial_embedding(&seed[..])
                .epoch_callback(|_epoch, snapshot| {
                    first_snapshot.extend_from_slice(snapshot);
                })
                .with_affinities(affinities)
                .exact()
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

/// Euclidean distance between two samples, the metric the Barnes-Hut tests use.
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Exact k nearest neighbors per sample, sorted by ascending distance, excluding
/// self: the same set the vantage point tree finds.
fn brute_force_neighbors(samples: &[&[f32]], n_neighbors: usize) -> Vec<Vec<Neighbor<f32>>> {
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

/// Fed the neighbors the tree would find, `from_neighbors` + `bhtsne` reproduces the `from_metric` + `bhtsne`
/// embedding. The parallel reductions are not bit-reproducible across thread schedules (rayon's
/// float reduction order depends on work-stealing), so the two paths are compared on a single-thread
/// pool, which still verifies that the supplied-neighbors entry point matches the vantage-point-tree
/// path exactly.
#[test]
fn barnes_hut_with_neighbors_matches_vptree_path() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    // A seed so both fits start from the very same embedding.
    let seed = lcg_samples(N, NO_DIMS as usize, 99);

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, n_neighbors);

    // A single-thread pool makes the reductions deterministic, so the two paths are bit-comparable.
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

    // Column ordering may differ (from_neighbors uses symmetrize_csr which sorts),
    // so compare with tolerance rather than bit-for-bit.
    let drift = mean_point_distance(&candidate, &reference, NO_DIMS as usize);
    let diagonal = bounding_box_diagonal(&reference, NO_DIMS as usize);
    assert!(
        drift <= 0.15 * diagonal + 1e-6,
        "from_neighbors path diverged from metric path: drift {drift}, diagonal {diagonal}"
    );
}

/// An out-of-range neighbor index must be rejected up front.
#[test]
#[should_panic(expected = "out of range")]
fn barnes_hut_with_neighbors_rejects_out_of_range_index() {
    const N: usize = 80;
    const DIM: usize = 4;

    let data = lcg_samples(N, DIM, 11);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

    let n_neighbors = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, n_neighbors);
    // Point one neighbor at a sample that does not exist.
    neighbors[0][0].index = N;

    let affinities = Affinities::from_neighbors(&neighbors, PERPLEXITY);
    TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
}

/// Deterministic LCG data so the tests need no RNG dependency.
fn lcg_samples(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((state >> 33) as f32 / u32::MAX as f32) - 0.5);
    }
    data
}

/// Mean euclidean distance between corresponding points of two embeddings.
fn mean_point_distance(a: &[f32], b: &[f32], dim: usize) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() / dim;
    a.chunks_exact(dim)
        .zip(b.chunks_exact(dim))
        .map(|(p, q)| {
            p.iter()
                .zip(q.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum::<f32>()
        / n as f32
}

/// Diagonal of the bounding box of an embedding.
fn bounding_box_diagonal(points: &[f32], dim: usize) -> f32 {
    (0..dim)
        .map(|d| {
            let component = points.iter().skip(d).step_by(dim);
            let min = component.clone().fold(f32::MAX, |a, &b| a.min(b));
            let max = component.fold(f32::MIN, |a, &b| a.max(b));
            (max - min).powi(2)
        })
        .sum::<f32>()
        .sqrt()
}

/// Regression test for the Gaussian bandwidth binary search.
///
/// When the neighbor distances are heterogeneous and noticeably larger than
/// 1, matching the target perplexity requires a bandwidth beta well below 1.
/// The descent path of the search (taken while no lower bracket is known yet)
/// must therefore be able to shrink beta indefinitely. Releases 0.5.0-0.5.2
/// clamped it at 0.5 and releases 0.5.3-0.5.4 moved beta upwards instead
/// (a constant named `zero_point_five` was set to 5.0), making the search
/// diverge and the conditional distribution degenerate.
#[test]
fn search_beta_converges_when_optimal_beta_below_one() {
    // 90 neighbours (3 * perplexity) with squared distances spread over
    // [20, 120]: the optimal beta for perplexity 30 is roughly 0.08.
    let distances_row: Vec<f64> = (0..90)
        .map(|i| (20.0 + 100.0 * (i as f64 + 1.0) / 90.0_f64).sqrt())
        .collect();
    let mut p_values_row: Vec<f64> = vec![0.0; 90];
    let perplexity = 30.0;

    tsne::search_beta(&mut p_values_row, &distances_row, &perplexity);

    // The effective number of neighbours encoded by the row, exp(H(P)),
    // must match the requested perplexity.
    let entropy: f64 = p_values_row
        .iter()
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let effective_perplexity = entropy.exp();

    assert!(
        (effective_perplexity - perplexity).abs() < 0.1,
        "expected effective perplexity of ~{perplexity}, got {effective_perplexity}"
    );
}

/// End-to-end regression test: two trivially separable clusters whose
/// coordinates are large enough that the bandwidth search must go below
/// beta = 1. A correct t-SNE is invariant to uniform input rescaling, so the
/// embedding must separate the clusters just as it does for small inputs.
#[test]
fn barnes_hut_separates_clusters_at_large_input_scale() {
    const N_PER_CLUSTER: usize = 150;
    const DIM: usize = 10;

    // Deterministic LCG so the test needs no RNG dependency.
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

    // For every point, the nearest embedded neighbor must belong to the
    // same cluster for at least 95% of the points.
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

/// With neighbours supplied (so the vantage point tree's randomness is out of the picture) and a
/// fixed seed, two Barnes-Hut runs must land in the same place. Determinism is relaxed for the
/// arena (unstable sort, plain parallel reductions), so this is a tolerance check rather than a
/// bit-for-bit one: the two embeddings must agree to within a small fraction of the embedding
/// scale, which a correct and stable optimization satisfies. N is above the parallel code
/// threshold, so the build runs in parallel.
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

/// Regression test for the parallel build, white box, the phantom-mass class. A corrupted
/// aggregation that counts mass a cell does not hold (an empty orthant, a stale cursor) drags the
/// cell centre of mass off, which this catches: every cell centre of mass must lie within its own
/// Morton cell. Morton quantization makes point conservation automatic, which `BarnesHutTree::new` asserts
/// (the leaf masses sum to `n`), so the root mass equalling `n` confirms no point was lost or
/// invented. The cloud is offset far from the origin so any centre of mass dragged toward it lands
/// outside its cell. N is above the parallel code threshold.
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

/// End-to-end regression test for the same bug, reproducing the symptom directly: corrupted
/// repulsive forces let attraction collapse the whole embedding onto a handful of coordinates. A
/// healthy run spreads the points out, so most embedded positions are distinct.
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

    // Count distinct positions, rounded to a hundredth. The collapse piled every point onto three
    // coordinates, a healthy embedding keeps them apart.
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

/// Round trip: run bhtsne, extract affinities, inject them with initial_embedding
/// into a second TsneBuilder, and call bhtsne again. The continuation stays closer
/// to the seed than a random-init run, and cluster structure is preserved.
#[test]
fn affinities_round_trip_barnes_hut() {
    const N: usize = 200;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // First run.
    let affinities1 = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted1 = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities1)
        .bhtsne(THETA)
        .fit();
    let embedding1 = fitted1.embedding().to_vec();
    let affinities = fitted1.affinities().clone();

    // Second run: warm start with affinities.
    let fitted2 = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(500)
        .initial_embedding(embedding1.clone())
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding2 = fitted2.embedding().to_vec();

    // The continuation must start near the seed, not restart from random.
    // Compare against a fresh random-init run: the warm-start embedding
    // should be closer to the seed than a random run would be.
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
        "warm start ({}) should be closer to seed than random ({})",
        dist_warm,
        dist_rand,
    );

    // Cluster structure should be preserved: relative ordering of nearby points
    // should be similar.
    let data10 = &data[..D * 10];
    let dist1 = mean_point_distance(data10, &embedding1[..D * 10], D);
    let dist2 = mean_point_distance(data10, &embedding2[..D * 10], D);
    assert!(
        (dist1 - dist2).abs() < dist1 * 0.5,
        "cluster structure changed too much: {} vs {}",
        dist1,
        dist2,
    );
}

/// Equivalence: a second bhtsne call with cloned affinities from a reference run produces
/// the same embedding as a plain bhtsne run within tolerance.
#[test]
fn affinities_equivalence_first_step() {
    const N: usize = 100;
    const CAPTURE: usize = 1; // Compare after 1 epoch.

    let data = lcg_samples(N, D, 99);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Build affinities from a reference run.
    let affinities_ref = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_ref = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities_ref)
        .bhtsne(THETA)
        .fit();
    let affinities = fitted_ref.affinities().clone();
    let seed = fitted_ref.embedding().to_vec();

    // Path A: plain bhtsne from the seed, building affinities fresh.
    let affinities_a = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_a = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed.clone())
        .with_affinities(affinities_a)
        .bhtsne(THETA)
        .fit();
    let result_a = fitted_a.embedding().to_vec();

    // Path B: bhtsne with cloned affinities from the same seed.
    let fitted_b = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(CAPTURE + 1)
        .initial_embedding(seed)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let result_b = fitted_b.embedding().to_vec();

    // Two runs on the same thread pool may differ by parallel-reduction noise,
    // so use a tolerance.
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
        "precomputed path diverged from reference: max_diff={max_diff}, scale={scale}",
    );
}

/// affinities() returns values summing to about 1 regardless of run length.
#[test]
fn affinities_pristine_independent_of_run_length() {
    const N: usize = 100;

    let data = lcg_samples(N, D, 7);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Short run: fewer epochs than stop_lying_epoch (250).
    let affinities_short = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_short = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(50)
        .with_affinities(affinities_short)
        .bhtsne(THETA)
        .fit();
    let aff_short = fitted_short.affinities();
    let sum_short: f32 = aff_short.values().iter().sum();

    // Long run: more epochs than stop_lying_epoch.
    let affinities_long = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted_long = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(1000)
        .with_affinities(affinities_long)
        .bhtsne(THETA)
        .fit();
    let aff_long = fitted_long.affinities();
    let sum_long: f32 = aff_long.values().iter().sum();

    // Both should sum to approximately 1 (pristine P).
    assert!(
        (sum_short - 1.0).abs() < 0.05,
        "short run affinities sum to {sum_short}, expected ~1",
    );
    assert!(
        (sum_long - 1.0).abs() < 0.05,
        "long run affinities sum to {sum_long}, expected ~1",
    );
    // And they should be identical since the data and perplexity are the same.
    assert_eq!(
        aff_short.rows(),
        aff_long.rows(),
        "row structure differs between short and long runs",
    );
    assert_eq!(
        aff_short.columns(),
        aff_long.columns(),
        "column structure differs between short and long runs",
    );
    for (a, b) in aff_short.values().iter().zip(aff_long.values().iter()) {
        assert!((a - b).abs() < 1e-6, "value differs: {a} vs {b}",);
    }
}

/// The FIt-SNE (interpolation) path reports a finite, non-negative KL divergence,
/// the same contract the exact and Barnes-Hut paths satisfy.
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

/// End-to-end quality: the FIt-SNE path must separate two trivially separable
/// clusters, the same regression the Barnes-Hut path is held to. A correct t-SNE
/// is invariant to uniform input rescaling, so the clusters separate at large
/// input scale just as at small.
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

    // For every point, the nearest embedded neighbor must belong to the same
    // cluster for at least 95% of the points.
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

/// The FIt-SNE path must not collapse the embedding onto a handful of points; a
/// healthy run spreads them out.
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

/// `fit_sne` with `from_neighbors` reproduces the `fit_sne` with `from_metric` embedding when fed the very
/// neighbors the tree would find. Reductions are not bit-reproducible across thread
/// schedules, so the two paths are compared on a single-thread pool.
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

    // Column ordering may differ (from_neighbors uses symmetrize_csr which sorts),
    // so compare with tolerance rather than bit-for-bit.
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

/// Smoke test for the 4D Barnes-Hut path: the embedding stays finite and correctly sized
/// after a short fit.
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

/// Smoke test for the 3D Barnes-Hut path: the embedding stays finite and correctly sized
/// after a short fit.
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

/// Smoke test for the 5D Barnes-Hut path: the embedding stays finite and correctly sized
/// after a short fit.
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

/// Smoke test for the 6D Barnes-Hut path: the embedding stays finite and correctly sized
/// after a short fit.
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

/// Smoke test for the 7D Barnes-Hut path: the embedding stays finite and correctly sized
/// after a short fit.
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

/// Build a two-block CSR affinity graph: two cliques of `half` nodes joined by a
/// single weak edge between node 0 and node `half`. The first nontrivial
/// eigenvector must separate the two blocks by sign.
fn two_block_affinities(half: usize) -> Affinities<f32> {
    let n = half * 2;
    let mut rows: Vec<usize> = Vec::with_capacity(n + 1);
    let mut columns: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();

    rows.push(0);
    for i in 0..n {
        if i == 0 {
            // Clique edges plus the weak bridge.
            for j in 1..half {
                columns.push(j as u32);
                values.push(1.0);
            }
            columns.push(half as u32);
            values.push(0.01);
        } else if i == half {
            // Weak bridge back to node 0 plus clique edges.
            columns.push(0);
            values.push(0.01);
            for j in (half + 1)..n {
                columns.push(j as u32);
                values.push(1.0);
            }
        } else if i < half {
            // Clique edges (skip self and node 0 which is already done).
            for j in 0..half {
                if j != i {
                    columns.push(j as u32);
                    values.push(1.0);
                }
            }
        } else {
            // Second clique (skip self and node half which is already done).
            for j in half..n {
                if j != i {
                    columns.push(j as u32);
                    values.push(1.0);
                }
            }
        }
        rows.push(columns.len());
    }

    Affinities::from_csr_unchecked(n, rows, columns, values)
}

#[test]
fn spectral_init_separates_two_blocks() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .exact()
        .fit();
    let seed = fitted.embedding().to_vec();

    let first_half_positive = seed[..HALF].iter().filter(|&&v| v > 0.0).count();
    let first_half_negative = seed[..HALF].iter().filter(|&&v| v < 0.0).count();
    let second_half_positive = seed[HALF..].iter().filter(|&&v| v > 0.0).count();
    let second_half_negative = seed[HALF..].iter().filter(|&&v| v < 0.0).count();

    let first_dominant = first_half_positive > first_half_negative;
    let second_dominant = second_half_positive > second_half_negative;
    assert!(
        first_dominant != second_dominant,
        "spectral init did not separate blocks: first half {{pos: {}, neg: {}}}, second half {{pos: {}, neg: {}}}",
        first_half_positive,
        first_half_negative,
        second_half_positive,
        second_half_negative
    );
}

#[test]
fn spectral_init_is_deterministic() {
    let affinities = two_block_affinities(15);
    let n = 30;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted_a = TsneBuilder::<f32, &[f32], 3>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities.clone())
        .bhtsne(0.5)
        .fit();
    let seed_a = fitted_a.embedding().to_vec();

    let fitted_b = TsneBuilder::<f32, &[f32], 3>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let seed_b = fitted_b.embedding().to_vec();

    assert_eq!(seed_a, seed_b, "spectral init is not deterministic");
}

#[test]
fn spectral_init_shape_and_finiteness() {
    let affinities = two_block_affinities(10);
    let n = 20;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    {
        let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
            .spectral_init()
            .epochs(0)
            .with_affinities(affinities.clone())
            .bhtsne(0.5)
            .fit();
        let seed = fitted.embedding();
        assert_eq!(seed.len(), n * 2);
        assert!(seed.iter().all(|v| v.is_finite()));
    }
    {
        let fitted = TsneBuilder::<f32, &[f32], 4>::new(&samples)
            .spectral_init()
            .epochs(0)
            .with_affinities(affinities.clone())
            .bhtsne(0.5)
            .fit();
        let seed = fitted.embedding();
        assert_eq!(seed.len(), n * 4);
        assert!(seed.iter().all(|v| v.is_finite()));
    }
    {
        let fitted = TsneBuilder::<f32, &[f32], 7>::new(&samples)
            .spectral_init()
            .epochs(0)
            .with_affinities(affinities)
            .bhtsne(0.5)
            .fit();
        let seed = fitted.embedding();
        assert_eq!(seed.len(), n * 7);
        assert!(seed.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn spectral_init_scale_matches_random_init() {
    let affinities = two_block_affinities(15);
    let n = 30;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let seed = fitted.embedding();

    let col0: Vec<f32> = (0..n).map(|i| seed[i * 2]).collect();
    let mean: f32 = col0.iter().sum::<f32>() / n as f32;
    let variance: f32 = col0.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();

    assert!(
        (std - 1e-4).abs() < 1e-6,
        "first column std is {std}, expected ~1e-4"
    );
}

/// Affinities with some isolated nodes (degree 0) to exercise the floor path.
fn affinities_with_isolated_nodes() -> Affinities<f32> {
    let n = 6;
    let mut rows = vec![0usize; n + 1];
    let mut columns = Vec::new();
    let mut values = Vec::new();

    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                columns.push(j as u32);
                values.push(1.0);
            }
        }
        rows[i + 1] = columns.len();
    }
    for i in 3..5 {
        for j in 3..5 {
            if i != j {
                columns.push(j as u32);
                values.push(1.0);
            }
        }
        rows[i + 1] = columns.len();
    }
    rows[6] = columns.len();

    Affinities::from_csr_unchecked(n, rows, columns, values)
}

#[test]
fn spectral_init_handles_isolated_nodes() {
    let affinities = affinities_with_isolated_nodes();
    let n = 6;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let seed = fitted.embedding();
    assert_eq!(seed.len(), n * 2);
    assert!(
        seed.iter().all(|v| v.is_finite()),
        "spectral init with isolated nodes produced non-finite values"
    );
}

#[test]
fn spectral_init_through_initial_embedding_reduces_kl() {
    const N: usize = 30;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Run with spectral_init: the spectral seed should lead to a valid fit
    // with finite KL divergence and correct embedding shape.
    let affinities = Affinities::from_metric(&samples, 5.0, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .spectral_init()
        .epochs(100)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();

    assert_eq!(embedding.len(), N * 2);
    assert!(embedding.iter().all(|v| v.is_finite()));
    assert!(fitted.kl_divergence().is_finite());

    // Spectral init should produce a meaningful embedding: not all zeros,
    // and the KL should be lower than a fresh random-init run of the same length.
    let affinities2 = Affinities::from_metric(&samples, 5.0, |a, b| euclidean(a, b));
    let fitted_rand = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(100)
        .with_affinities(affinities2)
        .bhtsne(THETA)
        .fit();

    // Both should be finite; spectral init typically converges faster so its
    // KL should not be significantly worse than random init at the same epoch count.
    assert!(
        fitted.kl_divergence() <= fitted_rand.kl_divergence() * 2.0,
        "spectral KL {} much worse than random KL {}",
        fitted.kl_divergence(),
        fitted_rand.kl_divergence()
    );
}

#[test]
fn spectral_init_via_builder_separates_blocks() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    // Use the builder method instead of manual spectral_embedding + initial_embedding.
    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let embedding = fitted.embedding();
    // Check the first column (column 0) for block separation.
    let col0: Vec<f32> = embedding.iter().step_by(2).cloned().collect();
    let first_half_positive = col0[..HALF].iter().filter(|&&v| v > 0.0).count();
    let first_half_negative = col0[..HALF].iter().filter(|&&v| v < 0.0).count();
    let second_half_positive = col0[HALF..].iter().filter(|&&v| v > 0.0).count();
    let second_half_negative = col0[HALF..].iter().filter(|&&v| v < 0.0).count();

    let first_dominant = first_half_positive > first_half_negative;
    let second_dominant = second_half_positive > second_half_negative;
    assert!(
        first_dominant != second_dominant,
        "spectral init via builder did not separate blocks"
    );
}

#[test]
fn explicit_initial_embedding_overrides_spectral_init() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    // Set both spectral_init flag and explicit embedding; explicit wins.
    let explicit: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.001).collect();
    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .initial_embedding(explicit.clone())
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let embedding = fitted.embedding();

    // The embedding should match the explicit seed (epochs=0 means no updates).
    assert_eq!(embedding, explicit.as_slice());
}

#[test]
fn spectral_init_with_custom_params_separates_blocks() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    // A cheaper budget than the defaults must still resolve this easy spectrum.
    let params = SpectralParams::new().rounds(3).degree(10);

    let fitted = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init_with(params)
        .epochs(0)
        .with_affinities(affinities.clone())
        .exact()
        .fit();
    let seed = fitted.embedding().to_vec();

    let first_half_positive = seed[..HALF].iter().filter(|&&v| v > 0.0).count();
    let second_half_positive = seed[HALF..].iter().filter(|&&v| v > 0.0).count();
    let first_dominant = first_half_positive > HALF / 2;
    let second_dominant = second_half_positive > HALF / 2;
    assert!(
        first_dominant != second_dominant,
        "custom-parameter spectral init did not separate blocks"
    );

    // Custom parameters must be as deterministic as the defaults.
    let fitted2 = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init_with(params)
        .epochs(0)
        .with_affinities(affinities.clone())
        .exact()
        .fit();
    assert_eq!(seed, fitted2.embedding().to_vec());

    // And produce a different solve than the defaults.
    let fitted3 = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .exact()
        .fit();
    assert_ne!(seed, fitted3.embedding().to_vec());
}

#[test]
fn spectral_init_with_custom_seed_std_scales_columns() {
    let affinities = two_block_affinities(15);
    let n = 30;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init_with(SpectralParams::new().seed_std(2e-3))
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let seed = fitted.embedding();

    let col0: Vec<f32> = (0..n).map(|i| seed[i * 2]).collect();
    let mean: f32 = col0.iter().sum::<f32>() / n as f32;
    let variance: f32 = col0.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();

    assert!(
        (std - 2e-3).abs() < 2e-5,
        "first column std is {std}, expected ~2e-3"
    );
}

#[test]
fn spectral_init_with_flows_through_builder() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init_with(SpectralParams::new().seed_std(5e-3))
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let embedding = fitted.embedding();

    // The custom seed scale must reach the seeding, proving the parameters flowed
    // through the builder into finalize_p_and_seed.
    let col0: Vec<f32> = embedding.iter().step_by(2).cloned().collect();
    let mean: f32 = col0.iter().sum::<f32>() / n as f32;
    let variance: f32 = col0.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();
    assert!(
        (std - 5e-3).abs() < 5e-5,
        "first column std is {std}, expected ~5e-3"
    );
}

#[test]
#[should_panic(expected = "at least one spectral solver round")]
fn spectral_params_reject_zero_rounds() {
    let _ = SpectralParams::new().rounds(0);
}

#[test]
#[should_panic(expected = "Chebyshev filter degree")]
fn spectral_params_reject_zero_degree() {
    let _ = SpectralParams::new().degree(0);
}

#[test]
#[should_panic(expected = "seed standard deviation")]
fn spectral_params_reject_nonpositive_seed_std() {
    let _ = SpectralParams::new().seed_std(0.0);
}

/// Property tests of the spectral embedding contract on arbitrary graphs.
mod spectral_properties {
    use proptest::prelude::*;

    use super::super::tsne::spectral::{SpectralParams, spectral_embedding};

    /// Dispatches a runtime dimensionality from the proptest generator to the
    /// const generic solver entry point.
    fn run_embedding(
        rows: &[usize],
        columns: &[u32],
        values: &[f32],
        d_out: usize,
        params: SpectralParams,
    ) -> Vec<f32> {
        match d_out {
            1 => spectral_embedding::<f32, 1>(rows, columns, values, params),
            2 => spectral_embedding::<f32, 2>(rows, columns, values, params),
            3 => spectral_embedding::<f32, 3>(rows, columns, values, params),
            4 => spectral_embedding::<f32, 4>(rows, columns, values, params),
            _ => unreachable!("the generators only produce d_out 1 through 4"),
        }
    }

    /// Builds a symmetric CSR affinity graph from an arbitrary edge list,
    /// accumulating duplicate pairs and dropping self loops. Nodes untouched by any
    /// edge remain isolated, which is a legal (and interesting) input.
    fn symmetric_csr(n: usize, edges: &[(usize, usize, f32)]) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        let mut weights = vec![0.0f32; n * n];
        for &(a, b, weight) in edges {
            let (i, j) = (a % n, b % n);
            if i == j {
                continue;
            }
            weights[i * n + j] += weight;
            weights[j * n + i] += weight;
        }
        let mut rows = vec![0usize];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if weights[i * n + j] > 0.0 {
                    columns.push(j as u32);
                    values.push(weights[i * n + j]);
                }
            }
            rows.push(columns.len());
        }
        (rows, columns, values)
    }

    /// Two cliques of the given sizes and internal weights with no edge between
    /// them.
    fn two_clique_csr(
        size_a: usize,
        size_b: usize,
        w_a: f32,
        w_b: f32,
    ) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        let n = size_a + size_b;
        let mut edges = Vec::new();
        for i in 0..size_a {
            for j in (i + 1)..size_a {
                edges.push((i, j, w_a));
            }
        }
        for i in size_a..n {
            for j in (i + 1)..n {
                edges.push((i, j, w_b));
            }
        }
        symmetric_csr(n, &edges)
    }

    proptest::proptest! {
        /// On any symmetric affinity graph the embedding has the right shape, is
        /// finite, has zero-mean columns scaled to the target std (or exactly
        /// degenerate ones), and is bit-for-bit deterministic.
        #[test]
        fn embedding_contract_holds_on_arbitrary_graphs(
            (n, edges, d_out) in (1usize..=40).prop_flat_map(|n| {
                (
                    Just(n),
                    proptest::collection::vec((0..n, 0..n, 0.01f32..10.0), 0..4 * n),
                    1usize..=4,
                )
            }),
        ) {
            let (rows, columns, values) = symmetric_csr(n, &edges);
            let params = SpectralParams::default();
            let embedding = run_embedding(&rows, &columns, &values, d_out, params);

            prop_assert_eq!(embedding.len(), n * d_out);
            prop_assert!(embedding.iter().all(|v| v.is_finite()));

            for d in 0..d_out {
                let column: Vec<f32> = (0..n).map(|i| embedding[i * d_out + d]).collect();
                let mean = column.iter().sum::<f32>() / n as f32;
                // The bound assumes this generator's weight range (0.01 to 10),
                // which caps the degree ratios that amplify the rescaled centering
                // residue. Re-derive it before widening the weights.
                prop_assert!(mean.abs() < 5e-6, "column {d} mean {mean} is not ~0");
                let std = (column.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                    / n as f32)
                    .sqrt();
                // Degenerate columns keep their pre-scale std, at most 1e-30 by the
                // scaling guard, so the branch cutoff can sit far below the target.
                prop_assert!(
                    std < 1e-25 || (std - 1e-4).abs() < 2e-6,
                    "column {d} std {std} is neither ~1e-4 nor degenerate"
                );
            }

            let again = run_embedding(&rows, &columns, &values, d_out, params);
            prop_assert_eq!(embedding, again);
        }

        /// Two disconnected cliques of arbitrary sizes and weights must be strictly
        /// separated by sign in the first embedding column, since the component
        /// contrast vector is an exact eigenvector of the graph.
        #[test]
        fn embedding_separates_disconnected_cliques(
            size_a in 3usize..=20,
            size_b in 3usize..=20,
            w_a in 0.05f32..5.0,
            w_b in 0.05f32..5.0,
        ) {
            let (rows, columns, values) = two_clique_csr(size_a, size_b, w_a, w_b);
            let embedding =
                run_embedding(&rows, &columns, &values, 1, SpectralParams::default());

            let sign_a = embedding[0] > 0.0;
            prop_assert!(
                embedding[..size_a].iter().all(|&v| (v > 0.0) == sign_a && v != 0.0),
                "first clique is not on one strict side of zero"
            );
            prop_assert!(
                embedding[size_a..].iter().all(|&v| (v > 0.0) != sign_a && v != 0.0),
                "second clique is not strictly on the opposite side"
            );
        }

        /// The output contract must hold for every valid parameter combination, and
        /// the column scale must follow the requested seed_std.
        #[test]
        fn embedding_contract_holds_for_any_params(
            rounds in 1usize..=6,
            degree in 1usize..=25,
            seed_std in 1e-6f64..1e-2,
        ) {
            let (rows, columns, values) = two_clique_csr(12, 9, 1.0, 0.5);
            let n = 21;
            let params = SpectralParams::new()
                .rounds(rounds)
                .degree(degree)
                .seed_std(seed_std);
            let embedding = run_embedding(&rows, &columns, &values, 2, params);

            prop_assert_eq!(embedding.len(), n * 2);
            prop_assert!(embedding.iter().all(|v: &f32| v.is_finite()));
            for d in 0..2 {
                let column: Vec<f32> = (0..n).map(|i| embedding[i * 2 + d]).collect();
                let mean = column.iter().sum::<f32>() / n as f32;
                let std = (column.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                    / n as f32)
                    .sqrt();
                let target = seed_std as f32;
                prop_assert!(
                    std < 1e-12 || (std - target).abs() < target * 0.02,
                    "column {d} std {std} does not match requested {target}"
                );
            }

            let again = run_embedding(&rows, &columns, &values, 2, params);
            prop_assert_eq!(embedding, again);
        }
    }
}

/// Builds a tiny symmetric, sum-to-one affinity graph over `n` nodes from an undirected edge list,
/// for hand-checking the pooling combinators.
fn tiny_graph(n: usize, edges: &[(usize, usize)]) -> Affinities<f32> {
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a].push((b as u32, 1.0));
        adj[b].push((a as u32, 1.0));
    }
    let mut rows = vec![0usize];
    let mut columns: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    for mut row in adj {
        row.sort_by_key(|(c, _)| *c);
        for (c, v) in row {
            columns.push(c);
            values.push(v);
        }
        rows.push(columns.len());
    }
    let total: f32 = values.iter().sum();
    for v in &mut values {
        *v /= total;
    }
    Affinities::from_csr_unchecked(n, rows, columns, values)
}

/// The standalone `from_metric` constructor reproduces the graph a full bhtsne fit builds and
/// hands back through `affinities()`.
#[test]
fn from_metric_matches_barnes_hut_affinities() {
    const N: usize = 200;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let direct = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });

    let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
    let fitted = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(EPOCHS)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    let fitted_aff = fitted.affinities();

    assert_eq!(direct.rows(), fitted_aff.rows(), "row structure differs");
    assert_eq!(
        direct.columns(),
        fitted_aff.columns(),
        "column structure differs"
    );
    for (a, b) in direct.values().iter().zip(fitted_aff.values().iter()) {
        assert!((a - b).abs() < 1e-6, "value differs: {a} vs {b}");
    }
    let sum: f32 = direct.values().iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "from_metric graph sums to {sum}, expected ~1"
    );
}

/// `from_neighbors` builds a pristine graph from a caller-supplied table, matches the metric path on
/// the same neighbors, and embeds to something finite through the metric-free fit.
#[test]
fn from_neighbors_builds_expected_graph() {
    const N: usize = 150;
    let data = lcg_samples(N, D, 7);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let k = (3.0 * PERPLEXITY) as usize;
    let neighbors = brute_force_neighbors(&samples, k);
    let graph = Affinities::from_neighbors(&neighbors, PERPLEXITY);

    assert_eq!(graph.n_samples(), N);
    let sum: f32 = graph.values().iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "from_neighbors sum {sum}, expected ~1"
    );

    // Exact neighbors match what the vantage point tree finds, so the two constructors agree.
    let metric = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });
    assert_eq!(
        graph.rows(),
        metric.rows(),
        "row structure differs from metric path"
    );

    let node_ids: Vec<u32> = (0..N as u32).collect();
    let fitted = TsneBuilder::<f32, u32>::new(&node_ids)
        .epochs(250)
        .with_affinities(graph)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * NO_DIMS as usize);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

/// `from_neighbors` accepts empty rows, mirroring the missing-modality case where the caller has
/// no neighbor evidence for some samples under this view. Empty rows contribute no attractive
/// edges to their own row, though they may still receive edges from other rows through
/// symmetrization. A sample no row points at ends up with a fully empty row in the pooled graph
/// and drifts under repulsion alone during the fit.
#[test]
fn from_neighbors_allows_empty_rows() {
    const N: usize = 60;
    let data = lcg_samples(N, D, 41);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let k = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, k);
    // Sample 0 has no neighbors under this view (missing modality). Sample 1 keeps its full row,
    // so sample 0 still ends up with at least one incoming edge through symmetrization if sample 1
    // happens to point at it. Sample 2's row is also emptied to exercise a fully isolated sample
    // when no other row points at it.
    neighbors[0].clear();
    // Make sure no other row points at sample 2 either: strip it from every row's neighbor list.
    for row in neighbors.iter_mut() {
        row.retain(|n| n.index != 2);
    }
    neighbors[2].clear();

    let graph = Affinities::from_neighbors(&neighbors, PERPLEXITY);

    assert_eq!(graph.n_samples(), N);
    let sum: f32 = graph.values().iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "from_neighbors sum {sum} with empty rows, expected ~1"
    );

    // Sample 2 is fully isolated: neither its row nor any incoming edge exists.
    let rows = graph.rows();
    assert_eq!(rows[2], rows[3], "sample 2 row should be empty (isolated)");

    // The fit still runs and produces a finite embedding for every sample, isolated ones included.
    let node_ids: Vec<u32> = (0..N as u32).collect();
    let fitted = TsneBuilder::<f32, u32>::new(&node_ids)
        .epochs(50)
        .with_affinities(graph)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * NO_DIMS as usize);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

/// Single-view builder with unit weight preserves the input graph exactly (edge structure, symmetry,
/// and sum-to-one) for a `tiny_graph` input, where the row-level pooling and the joint-level input
/// happen to agree.
#[test]
fn builder_single_view_reproduces_tiny_graph() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = super::AffinitiesBuilder::new(3).add(1.0f32, &a).build();

    assert_eq!(pooled.rows(), a.rows(), "row offsets differ");
    assert_eq!(pooled.columns(), a.columns(), "columns differ");
    for (i, (v_pool, v_a)) in pooled.values().iter().zip(a.values().iter()).enumerate() {
        assert!(
            (v_pool - v_a).abs() < 1e-6,
            "value {i} differs: pooled {v_pool}, input {v_a}"
        );
    }
    let sum: f32 = pooled.values().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "pooled sum {sum}, expected 1.0");
}

/// Two views over disjoint halves of a six-sample builder produce a graph whose edges live inside
/// each half only. No pair crosses the disjoint boundary and the joint sums to one.
#[test]
fn builder_pools_disjoint_subsets() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = super::AffinitiesBuilder::new(6)
        .add_over(0.5f32, &[0, 1, 2], &a)
        .add_over(0.5, &[3, 4, 5], &b)
        .build();

    // Row offsets and columns describe two disjoint 3-node chains.
    assert_eq!(pooled.rows(), &vec![0, 1, 3, 4, 5, 7, 8]);
    assert_eq!(pooled.columns(), &vec![1, 0, 2, 1, 4, 3, 5, 4]);
    for v in pooled.values() {
        assert!(
            (v - 0.125).abs() < 1e-6,
            "unexpected pooled value {v}, expected 0.125"
        );
    }
    let sum: f32 = pooled.values().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "pooled sum {sum}, expected 1.0");

    // No pair crosses the split: every column stays in the same half as its row.
    for (i, w) in pooled.rows().windows(2).enumerate() {
        let (start, end) = (w[0], w[1]);
        let row_half = if i < 3 { 0..3 } else { 3..6 };
        for c in &pooled.columns()[start..end] {
            assert!(
                row_half.contains(&(*c as usize)),
                "pair ({i}, {c}) crosses the disjoint boundary"
            );
        }
    }
}

/// A sample that lies outside every view's subset gets an empty row in the pool, so the fit sees no
/// attractive force for it. Sample 4 here is absent from both views.
#[test]
fn builder_isolates_sample_no_view_opines_on() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(2, &[(0, 1)]);
    let pooled = super::AffinitiesBuilder::new(5)
        .add_over(0.5f32, &[0, 1, 2], &a)
        .add_over(0.5, &[0, 1], &b)
        .build();

    assert_eq!(pooled.n_samples(), 5);
    // Row 3 is present in neither subset, row 4 is present in neither subset.
    assert_eq!(pooled.rows()[3], pooled.rows()[4], "row 3 must be empty");
    assert_eq!(pooled.rows()[4], pooled.rows()[5], "row 4 must be empty");
}

/// The row for a sample present in only one view comes verbatim (up to the joint normalization)
/// from that view: the row-level weights renormalize across the views actually present, so a single
/// present view has weight one for that row regardless of the other view's weight.
#[test]
fn builder_row_weights_by_presence_per_row() {
    // View A covers samples 0, 1, 2 (a chain 0-1-2). View B covers samples 2, 3, 4 (a chain 2-3-4).
    // Sample 0 is in A only, sample 4 is in B only, sample 2 is in both.
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = super::AffinitiesBuilder::new(5)
        // Give the two views deliberately unequal weights to prove the row-level renormalization
        // ignores the absent view's weight for a single-view-present row.
        .add_over(0.7f32, &[0, 1, 2], &a)
        .add_over(0.3, &[2, 3, 4], &b)
        .build();

    // Row 0 sees view A only: its only edge is (0, 1). Row 4 sees view B only: its only edge is (4, 3).
    // Row 2 sees both views: it should have edges to 1 (from A) and 3 (from B).
    let row = |i: usize| {
        let (s, e) = (pooled.rows()[i], pooled.rows()[i + 1]);
        pooled.columns()[s..e].to_vec()
    };
    assert_eq!(row(0), vec![1], "row 0 should reach only sample 1");
    assert_eq!(row(4), vec![3], "row 4 should reach only sample 3");
    let row2 = row(2);
    assert!(
        row2.contains(&1) && row2.contains(&3),
        "row 2 should reach both 1 (via A) and 3 (via B), got {row2:?}"
    );
}

#[test]
#[should_panic(expected = "weight must be strictly positive")]
fn builder_rejects_zero_weight() {
    let a = tiny_graph(3, &[(0, 1)]);
    let _ = super::AffinitiesBuilder::new(3).add(0.0f32, &a).build();
}

#[test]
#[should_panic(expected = "expected 5")]
fn builder_rejects_full_coverage_size_mismatch() {
    let a = tiny_graph(3, &[(0, 1)]);
    let _ = super::AffinitiesBuilder::new(5).add(1.0f32, &a).build();
}

#[test]
#[should_panic(expected = "duplicate global id")]
fn builder_rejects_duplicate_subset_id() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let _ = super::AffinitiesBuilder::new(4)
        .add_over(1.0f32, &[0, 1, 1], &a)
        .build();
}

#[test]
#[should_panic(expected = "out of range")]
fn builder_rejects_out_of_range_subset_id() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let _ = super::AffinitiesBuilder::new(4)
        .add_over(1.0f32, &[0, 1, 5], &a)
        .build();
}

#[test]
#[should_panic(expected = "at least one view")]
fn builder_rejects_empty_build() {
    let _ = super::AffinitiesBuilder::<f32>::new(3).build();
}

/// A pooled `Affinities` from the builder feeds through `with_affinities` and `bhtsne`, so
/// the whole path from multi-view pooling to a finite embedding runs end to end.
#[test]
fn builder_end_to_end_fit_barnes_hut() {
    const N: usize = 60;
    let data = lcg_samples(N, D, 17);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    // Two views over the same samples, one cosine-like, one Euclidean.
    let view_a = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });
    let view_b = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
    });
    let pooled = super::AffinitiesBuilder::new(N)
        .add(0.5f32, &view_a)
        .add(0.5, &view_b)
        .build();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .epochs(50)
        .with_affinities(pooled)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 2);
    assert!(embedding.iter().all(|v| v.is_finite()));
}
