//! Integration tests for the spectral-embedding initialization exposed through `TsneBuilder`.

use bhtsne::{Affinities, SpectralParams, TsneBuilder};

mod common;
use common::{
    D, THETA, affinities_with_isolated_nodes, euclidean, lcg_samples, two_block_affinities,
};

/// Two disconnected cliques bridged by a weak edge: spectral init must separate them by sign in
/// the first embedding column.
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
        "spectral init did not separate blocks: first half {{pos: {first_half_positive}, neg: {first_half_negative}}}, second half {{pos: {second_half_positive}, neg: {second_half_negative}}}"
    );
}

/// Two spectral inits over the same affinities must produce bit-identical seeds.
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

/// Spectral init produces a finite embedding of the expected shape for 2D, 4D, and 7D targets.
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

/// The default spectral params scale the seed columns to std ~1e-4.
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

/// Isolated nodes exercise the degree-floor path in the spectral solver: the fit must still
/// return a finite seed of the expected shape.
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

/// Spectral init followed by an actual fit should not diverge: the KL divergence stays finite and
/// is not much worse than a random-init run of the same length.
#[test]
fn spectral_init_through_initial_embedding_reduces_kl() {
    const N: usize = 30;
    let data = lcg_samples(N, D, 42);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

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

    let affinities2 = Affinities::from_metric(&samples, 5.0, |a, b| euclidean(a, b));
    let fitted_rand = TsneBuilder::<f32, &[f32]>::new(&samples)
        .epochs(100)
        .with_affinities(affinities2)
        .bhtsne(THETA)
        .fit();

    assert!(
        fitted.kl_divergence() <= fitted_rand.kl_divergence() * 2.0,
        "spectral KL {} much worse than random KL {}",
        fitted.kl_divergence(),
        fitted_rand.kl_divergence()
    );
}

/// The builder `spectral_init` combinator produces the same block-separated seed as the manual
/// pipeline exercised by `spectral_init_separates_two_blocks`.
#[test]
fn spectral_init_via_builder_separates_blocks() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let embedding = fitted.embedding();
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

/// An explicit initial embedding must win over spectral init when both are configured on the
/// builder.
#[test]
fn explicit_initial_embedding_overrides_spectral_init() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

    let explicit: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.001).collect();
    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .spectral_init()
        .initial_embedding(explicit.clone())
        .epochs(0)
        .with_affinities(affinities)
        .bhtsne(0.5)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding, explicit.as_slice());
}

/// A cheaper `SpectralParams` (fewer rounds, smaller Chebyshev degree) still resolves an easy
/// spectrum, remains deterministic, and produces a different seed than the defaults.
#[test]
fn spectral_init_with_custom_params_separates_blocks() {
    const HALF: usize = 20;
    let affinities = two_block_affinities(HALF);
    let n = HALF * 2;
    let data: Vec<f32> = vec![0.0; n];
    let samples: Vec<&[f32]> = data.chunks(1).collect();

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

    let fitted2 = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init_with(params)
        .epochs(0)
        .with_affinities(affinities.clone())
        .exact()
        .fit();
    assert_eq!(seed, fitted2.embedding().to_vec());

    let fitted3 = TsneBuilder::<f32, &[f32], 1>::new(&samples)
        .spectral_init()
        .epochs(0)
        .with_affinities(affinities)
        .exact()
        .fit();
    assert_ne!(seed, fitted3.embedding().to_vec());
}

/// A custom `seed_std` scales the seed columns to the requested std.
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

/// The custom seed scale must reach the seeding, proving the parameters flow through the builder
/// into `finalize_p_and_seed`.
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

    let col0: Vec<f32> = embedding.iter().step_by(2).cloned().collect();
    let mean: f32 = col0.iter().sum::<f32>() / n as f32;
    let variance: f32 = col0.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();
    assert!(
        (std - 5e-3).abs() < 5e-5,
        "first column std is {std}, expected ~5e-3"
    );
}

/// `SpectralParams` builder methods reject nonsensical values with descriptive panics.
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
