//! Integration tests for the `AffinitiesBuilder` pool + agreement pool, plus the fit-integration
//! round-trip tests that mix affinity constructors with a full bhtsne fit.

use bhtsne::error::AffinitiesBuilderError as Err;
use bhtsne::{Affinities, AffinitiesBuilder, TsneBuilder};

mod common;
use common::{
    D, EPOCHS, NO_DIMS, PERPLEXITY, THETA, brute_force_neighbors, euclidean, lcg_samples,
    tiny_graph,
};

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

/// `from_neighbors` builds a pristine graph from a caller-supplied table, matches the metric path
/// on the same neighbors, and embeds to a finite value through the metric-free fit.
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

/// `from_neighbors` accepts empty rows: the fit still runs and produces a finite embedding for
/// every sample, isolated ones included.
#[test]
fn from_neighbors_allows_empty_rows() {
    const N: usize = 60;
    let data = lcg_samples(N, D, 41);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let k = (3.0 * PERPLEXITY) as usize;
    let mut neighbors = brute_force_neighbors(&samples, k);
    neighbors[0].clear();
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

    let rows = graph.rows();
    assert_eq!(rows[2], rows[3], "sample 2 row should be empty (isolated)");

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

/// Single-view builder with unit weight preserves the input graph exactly (edge structure,
/// symmetry, sum-to-one) for a `tiny_graph` input where row-level pooling and joint-level input
/// happen to agree.
#[test]
fn builder_single_view_reproduces_tiny_graph() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = AffinitiesBuilder::new(3)
        .add(1.0f32, &a)
        .unwrap()
        .build()
        .unwrap();

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
    let pooled = AffinitiesBuilder::new(6)
        .add_over(0.5f32, &[0, 1, 2], &a)
        .unwrap()
        .add_over(0.5, &[3, 4, 5], &b)
        .unwrap()
        .build()
        .unwrap();

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

/// A sample that lies outside every view's subset gets an empty row in the pool.
#[test]
fn builder_isolates_sample_no_view_opines_on() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(2, &[(0, 1)]);
    let pooled = AffinitiesBuilder::new(5)
        .add_over(0.5f32, &[0, 1, 2], &a)
        .unwrap()
        .add_over(0.5, &[0, 1], &b)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pooled.n_samples(), 5);
    assert_eq!(pooled.rows()[3], pooled.rows()[4], "row 3 must be empty");
    assert_eq!(pooled.rows()[4], pooled.rows()[5], "row 4 must be empty");
}

/// Row-level renormalization: a sample present in only one view gets that view's contribution
/// with weight one, ignoring the absent view's weight.
#[test]
fn builder_row_weights_by_presence_per_row() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = AffinitiesBuilder::new(5)
        .add_over(0.7f32, &[0, 1, 2], &a)
        .unwrap()
        .add_over(0.3, &[2, 3, 4], &b)
        .unwrap()
        .build()
        .unwrap();

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
fn builder_rejects_zero_weight() {
    let a = tiny_graph(3, &[(0, 1)]);
    let e = AffinitiesBuilder::new(3).add(0.0f32, &a).err().unwrap();
    assert!(matches!(e, Err::NonPositiveWeight));
}

#[test]
fn builder_rejects_full_coverage_size_mismatch() {
    let a = tiny_graph(3, &[(0, 1)]);
    let e = AffinitiesBuilder::new(5).add(1.0f32, &a).err().unwrap();
    assert!(matches!(
        e,
        Err::SampleCountMismatch {
            view: 3,
            builder: 5
        }
    ));
}

#[test]
fn builder_rejects_duplicate_subset_id() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let e = AffinitiesBuilder::new(4)
        .add_over(1.0f32, &[0, 1, 1], &a)
        .err()
        .unwrap();
    assert!(matches!(e, Err::DuplicateSubsetId { id: 1 }));
}

#[test]
fn builder_rejects_out_of_range_subset_id() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let e = AffinitiesBuilder::new(4)
        .add_over(1.0f32, &[0, 1, 5], &a)
        .err()
        .unwrap();
    assert!(matches!(
        e,
        Err::SubsetIdOutOfRange {
            id: 5,
            n_samples: 4
        }
    ));
}

#[test]
fn builder_rejects_empty_build() {
    let e = AffinitiesBuilder::<f32>::new(3).build().err().unwrap();
    assert!(matches!(e, Err::NoViews));
}

/// A pooled `Affinities` from the builder feeds through `with_affinities` and `bhtsne`, so the
/// whole path from multi-view pooling to a finite embedding runs end to end.
#[test]
fn builder_end_to_end_fit_barnes_hut() {
    const N: usize = 60;
    let data = lcg_samples(N, D, 17);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let view_a = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });
    let view_b = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
    });
    let pooled = AffinitiesBuilder::new(N)
        .add(0.5f32, &view_a)
        .unwrap()
        .add(0.5, &view_b)
        .unwrap()
        .build()
        .unwrap();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .epochs(50)
        .with_affinities(pooled)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 2);
    assert!(embedding.iter().all(|v| v.is_finite()));
}

/// Agreement-weighted pool downweights a dissenting view per row: two views agree on short-range
/// chain edges, and view C picks long-range edges nobody else sees. Under softmax the C
/// contribution is negligible relative to the crowd.
#[test]
fn agreement_downweights_dissenting_view() {
    let a = tiny_graph(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]);
    let b = tiny_graph(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]);
    let c = tiny_graph(6, &[(0, 3), (1, 4), (2, 5)]);
    let pooled = AffinitiesBuilder::new(6)
        .add(1.0f32, &a)
        .unwrap()
        .add(1.0, &b)
        .unwrap()
        .add(1.0, &c)
        .unwrap()
        .build()
        .unwrap();

    let leaked_pairs = [(0u32, 3u32), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)];
    let ab_pairs = [(0u32, 1u32), (3, 4), (2, 1), (4, 3), (5, 4), (4, 5)];
    for ((row, col), (ab_row, ab_col)) in leaked_pairs.iter().zip(ab_pairs.iter()) {
        let (rs, re) = (
            pooled.rows()[*row as usize],
            pooled.rows()[*row as usize + 1],
        );
        let cols = &pooled.columns()[rs..re];
        let vals = &pooled.values()[rs..re];
        let c_val = cols
            .iter()
            .position(|c| c == col)
            .map(|i| vals[i])
            .unwrap_or(0.0);

        let (abs, abe) = (
            pooled.rows()[*ab_row as usize],
            pooled.rows()[*ab_row as usize + 1],
        );
        let ab_cols = &pooled.columns()[abs..abe];
        let ab_vals = &pooled.values()[abs..abe];
        let ab_val = ab_cols
            .iter()
            .position(|c| c == ab_col)
            .map(|i| ab_vals[i])
            .unwrap_or(0.0);

        assert!(
            ab_val > 0.0,
            "expected A/B edge ({ab_row}, {ab_col}) to be present"
        );
        assert!(
            c_val < 0.01 * ab_val,
            "C edge ({row}, {col}) carries {c_val} vs A/B edge {ab_val}; softmax did not suppress it"
        );
    }

    let sum: f32 = pooled.values().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "agreement-weighted pool sum {sum} != 1"
    );
}

/// When every present view completely disagrees with the crowd at some row, the row falls back to
/// the unit-weighted pool, matching plain `build_uniform()`.
#[test]
fn agreement_fallback_when_every_view_disagrees() {
    let a = tiny_graph(6, &[(0, 1), (2, 3), (4, 5)]);
    let b = tiny_graph(6, &[(0, 3), (1, 4), (2, 5)]);
    let c = tiny_graph(6, &[(0, 2), (1, 5), (3, 4)]);
    let pooled = AffinitiesBuilder::new(6)
        .add(1.0f32, &a)
        .unwrap()
        .add(1.0, &b)
        .unwrap()
        .add(1.0, &c)
        .unwrap()
        .build()
        .unwrap();
    let plain = AffinitiesBuilder::new(6)
        .add(1.0f32, &a)
        .unwrap()
        .add(1.0, &b)
        .unwrap()
        .add(1.0, &c)
        .unwrap()
        .build_uniform()
        .unwrap();

    assert_eq!(pooled.rows(), plain.rows(), "row structure diverged");
    assert_eq!(
        pooled.columns(),
        plain.columns(),
        "column structure diverged"
    );
    for (p, u) in pooled.values().iter().zip(plain.values()) {
        assert!(
            (p - u).abs() < 1e-6,
            "fallback pool differs from plain build: {p} vs {u}"
        );
    }
}

/// Subset views that never overlap at any global sample force the fallback per row and the
/// agreement pool reduces to plain `build_uniform()`.
#[test]
fn agreement_over_disjoint_subsets_falls_back_to_plain_build() {
    let a = tiny_graph(3, &[(0, 1), (1, 2)]);
    let b = tiny_graph(3, &[(0, 1), (1, 2)]);
    let c = tiny_graph(3, &[(0, 1), (1, 2)]);
    let pooled = AffinitiesBuilder::new(9)
        .add_over(1.0f32, &[0, 1, 2], &a)
        .unwrap()
        .add_over(1.0, &[3, 4, 5], &b)
        .unwrap()
        .add_over(1.0, &[6, 7, 8], &c)
        .unwrap()
        .build()
        .unwrap();
    let plain = AffinitiesBuilder::new(9)
        .add_over(1.0f32, &[0, 1, 2], &a)
        .unwrap()
        .add_over(1.0, &[3, 4, 5], &b)
        .unwrap()
        .add_over(1.0, &[6, 7, 8], &c)
        .unwrap()
        .build_uniform()
        .unwrap();

    assert_eq!(pooled.n_samples(), 9);
    assert_eq!(pooled.rows(), plain.rows());
    assert_eq!(pooled.columns(), plain.columns());
    for (p, u) in pooled.values().iter().zip(plain.values()) {
        assert!((p - u).abs() < 1e-6);
    }
}

/// End-to-end: three views (two agree, one dissents) pooled through `build` feed a Barnes-Hut fit
/// that produces a finite embedding of the right shape.
#[test]
fn agreement_end_to_end_fit() {
    const N: usize = 60;
    let data = lcg_samples(N, D, 23);
    let samples: Vec<&[f32]> = data.chunks(D).collect();

    let view_a = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });
    let view_b = Affinities::from_metric(&samples, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    });
    let noise_data = lcg_samples(N, D, 4242);
    let noise_rows: Vec<&[f32]> = noise_data.chunks(D).collect();
    let view_c = Affinities::from_metric(&noise_rows, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
        euclidean(a, b)
    });

    let pooled = AffinitiesBuilder::new(N)
        .add(1.0f32, &view_a)
        .unwrap()
        .add(1.0, &view_b)
        .unwrap()
        .add(1.0, &view_c)
        .unwrap()
        .build()
        .unwrap();

    let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
        .epochs(50)
        .with_affinities(pooled)
        .bhtsne(THETA)
        .fit();
    let embedding = fitted.embedding();
    assert_eq!(embedding.len(), N * 2);
    assert!(embedding.iter().all(|v| v.is_finite()));
}
