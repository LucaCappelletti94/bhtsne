//! Integration tests for the exact fit path plus the early-exaggeration knobs it exercises.

use bhtsne::{Affinities, TsneBuilder};

mod common;
use common::{
    D, EPOCHS, NO_DIMS, PERPLEXITY, THETA, bounding_box_diagonal, exact_snapshot_at, lcg_samples,
    mean_point_distance,
};

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

/// End-to-end iris fit through the exact solver. Requires the iris dataset to be present in the
/// working directory, so is ignored by default.
#[cfg(feature = "csv")]
#[test]
#[ignore = "requires iris dataset"]
fn exact_tsne() {
    let data: Vec<f32> =
        bhtsne::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
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

/// Same for the Barnes-Hut solver. Requires the iris dataset.
#[cfg(feature = "csv")]
#[test]
#[ignore = "requires iris dataset"]
fn barnes_hut_tsne() {
    let data: Vec<f32> =
        bhtsne::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();
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

/// Regression: `epoch_callback` is invoked once per epoch, in order, and the final snapshot must
/// match the value the exact fit returns.
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

/// The epoch callback is invoked only on the fitting thread, so it need not be `Send` or `Sync`.
/// A closure capturing an `Rc<RefCell<_>>` is neither, which the previous bound rejected. This
/// is exactly the shape a single-threaded wasm worker needs.
#[test]
fn epoch_callback_accepts_non_send_closure() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const N: usize = 40;
    const DIM: usize = 4;
    const RUN_EPOCHS: usize = 10;

    let data = lcg_samples(N, DIM, 7);
    let samples: Vec<&[f32]> = data.chunks(DIM).collect();

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

/// The exact-fit warm start begins from the supplied embedding: the first epoch stays close to
/// the seed, far closer than a random init near the origin would.
#[test]
fn warm_start_begins_from_initial_embedding_exact() {
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

    let origin = vec![0.0_f32; seed.len()];
    let random_displacement = mean_point_distance(&origin, &seed, dim);
    assert!(
        random_displacement > 10.0 * displacement,
        "warm start indistinguishable from a random initialization: {displacement} against {random_displacement}"
    );
}

/// The exact fit rejects an initial embedding whose length does not match `n_samples * D`.
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

/// A `stop_lying_epoch` of zero disables the early exaggeration factor for the exact fit as well.
#[test]
fn stop_lying_epoch_zero_skips_exaggeration_exact() {
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

/// Setting the factor to its default `12.0` explicitly must match leaving it unset: two runs
/// execute the same arithmetic and can differ only by parallel-reduction noise.
#[test]
fn early_exaggeration_explicit_twelve_matches_default() {
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

/// Two fits differing only in `early_exaggeration` pull the embedding apart by different amounts
/// in the early epochs, so their first-epoch snapshots are measurably different.
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

/// `early_exaggeration(1.0)` is a second way to say "no exaggeration": it must match
/// `stop_lying_epoch(0)`. Both leave the `P` distribution unexaggerated.
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
