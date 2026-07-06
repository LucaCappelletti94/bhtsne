<div align="center"> <h1 align="center"> bhtsne </h1> </div>

<div align="center">

[![CI](https://github.com/frjnn/bhtsne/actions/workflows/ci.yml/badge.svg)](https://github.com/frjnn/bhtsne/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/bhtsne.svg)](https://crates.io/crates/bhtsne)
[![docs.rs](https://docs.rs/bhtsne/badge.svg)](https://docs.rs/bhtsne)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/frjnn/bhtsne/branch/master/graph/badge.svg)](https://codecov.io/gh/frjnn/bhtsne)

</div>


Parallel Barnes-Hut and exact implementations of the t-SNE algorithm written in Rust. The tree-accelerated version of the algorithm is described in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The exact original version is described in [this paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by [G. Hinton](https://www.cs.toronto.edu/~hinton/) and Laurens van der Maaten. Additional implementations are listed at [this page](http://lvdmaaten.github.io/tsne/).

## Installation

```toml
[dependencies]
bhtsne = "0.8"
```

## Basic use

Build an `Affinities` graph, pass it through a fit builder, and read the embedding off the returned `Fitted*` result. `Affinities::from_l2` is sugar for the Euclidean metric; `from_cosine` is sugar for L2-normalized rows; `from_metric(&data, perplexity, metric)` accepts any `Fn(&U, &U) -> T` for custom types.

```rust
use bhtsne::{Affinities, TsneBuilder};

// Thirty points on two 3-dimensional blobs.
let raw: Vec<[f32; 3]> = (0..30)
    .map(|i| {
        let cluster = (i / 15) as f32;
        let angle = (i % 15) as f32 * 0.4;
        [cluster * 4.0 + angle.cos(), cluster * 4.0 + angle.sin(), 0.0]
    })
    .collect();
let samples: Vec<&[f32]> = raw.iter().map(|row| row.as_slice()).collect();

let affinities = Affinities::from_l2(&samples, 5.0_f32);
let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
    .epochs(250)
    .with_affinities(affinities)
    .bhtsne(0.5)
    .fit();

let embedding = fitted.embedding();
assert_eq!(embedding.len(), samples.len() * 2);
assert!(embedding.iter().all(|v| v.is_finite()));
```

The Barnes-Hut path (`.bhtsne(theta)`) supports embedding dimensionality `D` of 2, 3, 4, 5, 6, or 7 (32, 21, 16, 25, 21, and 18 bits per axis, respectively, in the Morton codec). The exact path (`.exact()`) stays general for any `D`. The FIt-SNE path (`.fit_sne()`) is restricted to `D` of 1 or 2 (the interpolation grid stays tractable) and is roughly flat in `n`, matching Barnes-Hut around 50k points and pulling ahead beyond, roughly 1.3x at 70k points and widening. It swaps in place of `.bhtsne(theta)` in the chain above.

The Morton (Z-order) linear tree the Barnes-Hut path walks lives in the sibling [`barnes-hut-tree`](barnes-hut-tree/README.md) crate, published standalone; any other force-approximation problem that needs a Z-order linear tree in a contiguous arena can pull it in directly through its `Arena`, `Morton`, and `Dim` API.

## Multi-view fusion

When one similarity does not capture the whole story (features on one side, a graph on the other, or several kernels over the same data), build each view separately as an `Affinities` and pool them through `AffinitiesBuilder`. The pool is a per-row convex combination, so a view that only opines on some of the samples contributes only to the rows it actually reaches, and the weights renormalize across the views present for each row.

```rust
use bhtsne::{Affinities, AffinitiesBuilder, TsneBuilder};

// Thirty samples with two independent 3-dimensional feature views over the same rows.
let view_a_raw: Vec<[f32; 3]> = (0..30)
    .map(|i| {
        let cluster = (i / 15) as f32;
        let angle = (i % 15) as f32 * 0.4;
        [cluster * 4.0 + angle.cos(), cluster * 4.0 + angle.sin(), 0.0]
    })
    .collect();
let view_b_raw: Vec<[f32; 3]> = (0..30)
    .map(|i| {
        let cluster = (i % 2) as f32;
        let t = (i as f32) * 0.3;
        [cluster * 3.0, t.sin(), t.cos()]
    })
    .collect();
let view_a_rows: Vec<&[f32]> = view_a_raw.iter().map(|r| r.as_slice()).collect();
let view_b_rows: Vec<&[f32]> = view_b_raw.iter().map(|r| r.as_slice()).collect();

let view_a = Affinities::from_l2(&view_a_rows, 5.0_f32);
let view_b = Affinities::from_l2(&view_b_rows, 5.0_f32);
let pooled = AffinitiesBuilder::new(view_a_rows.len())
    .add(0.5, &view_a)
    .add(0.5, &view_b)
    .build();

let node_ids: Vec<u32> = (0..view_a_rows.len() as u32).collect();
let fitted = TsneBuilder::<f32, u32, 2>::new(&node_ids)
    .epochs(250)
    .with_affinities(pooled)
    .bhtsne(0.5)
    .fit();

assert_eq!(fitted.embedding().len(), node_ids.len() * 2);
assert!(fitted.embedding().iter().all(|v| v.is_finite()));
```

## Partial views (missing modalities)

When features or similarities are only defined for a subset of the samples (missing modalities, partly disconnected graphs, one modality dropping out for some rows), build each view locally over its own samples and pool with `add_over(weight, &subset, &view)`. Each view carries its own local `0..k` indexing, and `subset[i]` names the global id of the view's `i`th local sample. Samples the view has no opinion on stay out of that view's row-normalized contribution; a sample no view opines on keeps an empty row in the pooled graph and drifts under repulsion alone during the fit.

The example below has forty global samples: the first twenty-five carry a feature vector, the last twenty-five carry a precomputed neighbor list arranged in a ring, so the middle ten samples appear in both modalities, the outer thirty appear in only one, and no sample is missing from both.

```rust
use bhtsne::{Affinities, AffinitiesBuilder, Neighbor, TsneBuilder};

const N: usize = 40;

// Feature view: samples 0..25 have a 3-dimensional feature vector; samples 25..40 do not.
let feature_raw: Vec<[f32; 3]> = (0..25)
    .map(|i| {
        let cluster = (i / 12) as f32;
        let angle = (i % 12) as f32 * 0.5;
        [cluster * 4.0 + angle.cos(), cluster * 4.0 + angle.sin(), 0.0]
    })
    .collect();
let feature_rows: Vec<&[f32]> = feature_raw.iter().map(|r| r.as_slice()).collect();
let featured: Vec<usize> = (0..25).collect();
let feature_view = Affinities::from_l2(&feature_rows, 5.0_f32);

// Graph view: samples 15..40 have a precomputed neighbor list arranged in a ring.
let neighbored: Vec<usize> = (15..40).collect();
let k = neighbored.len();
let mut neighbors: Vec<Vec<Neighbor<f32>>> = vec![Vec::new(); k];
for local in 0..k {
    let prev = (local + k - 1) % k;
    let next = (local + 1) % k;
    neighbors[local].push(Neighbor { index: prev, distance: 1.0 });
    neighbors[local].push(Neighbor { index: next, distance: 1.0 });
}
let graph_view = Affinities::from_neighbors(&neighbors, 5.0_f32);

let pooled = AffinitiesBuilder::new(N)
    .add_over(0.5, &featured, &feature_view)
    .add_over(0.5, &neighbored, &graph_view)
    .build();

let node_ids: Vec<u32> = (0..N as u32).collect();
let fitted = TsneBuilder::<f32, u32, 2>::new(&node_ids)
    .epochs(250)
    .with_affinities(pooled)
    .bhtsne(0.5)
    .fit();

assert_eq!(fitted.embedding().len(), N * 2);
assert!(fitted.embedding().iter().all(|v| v.is_finite()));
```

If you instead already hold a global-indexed neighbor table (one row per global sample) with natural missingness, `Affinities::from_neighbors` accepts empty rows directly, so you can pass the whole table through without splitting into a subset first: a sample whose row is empty gets no attractive edges of its own, still receives any incoming edges through symmetrization, and drifts under repulsion alone when neither side connects it.

See `bhtsne/examples/cora_tsne.rs` for a full walkthrough that pools a bounded shortest-path view over the Cora citation graph with a cosine view over the bag-of-words features.

## Parallelism

Built on [rayon](https://github.com/rayon-rs/rayon), the algorithm uses the current thread pool (defaults to one thread per logical core). See [rayon's FAQ](https://github.com/rayon-rs/rayon/blob/master/FAQ.md) for details.

## MNIST embedding

The embedding below was obtained by preprocessing the [MNIST](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) train set with PCA down to 20 dimensions. It takes about 20 seconds on a M5 MacBook Pro.

<p align="center">
  <img src="imgs/mnist_embedding.gif" alt="mnist" />
</p>
