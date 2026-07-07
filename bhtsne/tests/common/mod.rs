//! Shared fixtures for the bhtsne integration test binaries. Each `tests/*.rs` file compiles as
//! its own crate, so this module is included via `mod common;` from every test binary that
//! needs it. All items are `#[allow(dead_code)]` because different test binaries use different
//! subsets.

#![allow(dead_code)]

use bhtsne::{Affinities, Neighbor, TsneBuilder};

pub const D: usize = 4;
pub const THETA: f32 = 0.5;
pub const PERPLEXITY: f64 = 10.;
pub const EPOCHS: usize = 2_000;
pub const NO_DIMS: u8 = 2;

/// Deterministic pseudo random samples. LCG seeded on `state` so tests are reproducible.
pub fn lcg_samples(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
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
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Squared euclidean distance between two f32 samples.
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Exact `n_neighbors` nearest neighbors per sample, sorted ascending by distance and excluding
/// the sample itself.
pub fn brute_force_neighbors(samples: &[&[f32]], n_neighbors: usize) -> Vec<Vec<Neighbor<f32>>> {
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

/// Mean euclidean distance between corresponding points of two embeddings.
pub fn mean_point_distance(a: &[f32], b: &[f32], dim: usize) -> f32 {
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
pub fn bounding_box_diagonal(points: &[f32], dim: usize) -> f32 {
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

/// Runs a short exact fit from a fixed seed and returns the embedding snapshot at the end of
/// epoch `capture`, so two configurations can be compared at the same point of the optimization.
/// `configure` sets the knob under test on the builder.
pub fn exact_snapshot_at<F>(capture: usize, configure: F) -> Vec<f32>
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

/// Two disconnected cliques of size `half` bridged by a single weak edge. The Laplacian's second
/// smallest eigenvalue is tiny and its eigenvector separates the two blocks by sign, so the
/// spectral solver has to resolve a well-known but tightly-spaced pair.
pub fn two_block_affinities(half: usize) -> Affinities<f32> {
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
            for j in 0..half {
                if j != i {
                    columns.push(j as u32);
                    values.push(1.0);
                }
            }
        } else {
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

/// A tiny CSR affinity graph with two 3-clique components (0-1-2 and 3-4) and one fully isolated
/// node (5). Exercises the degree floor path of the spectral solver.
pub fn affinities_with_isolated_nodes() -> Affinities<f32> {
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

/// Builds a tiny symmetric, sum-to-one affinity graph over `n` nodes from an undirected edge list,
/// for hand-checking the pooling combinators.
pub fn tiny_graph(n: usize, edges: &[(usize, usize)]) -> Affinities<f32> {
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
