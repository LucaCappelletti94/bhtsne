# bhtsne API redesign

Working design doc for the "first-time-right" ergonomics pass. Not committed. Discardable once the code lands and matches this shape.

## Public surface (after the rewrite)

```rust
// Errors
pub enum FromCsrError { WrongRowsLen, NonMonotoneRows, ..., ZeroTotalMass }

// Affinities and constructors
pub struct Neighbor<T> { pub index: usize, pub distance: T }
pub struct Affinities<T> { rows: Vec<usize>, columns: Vec<u32>, values: Vec<T> }
impl<T> Affinities<T> {
    // Accessors
    pub fn n_samples(&self) -> usize;
    pub fn rows(&self) -> &[usize];
    pub fn columns(&self) -> &[u32];
    pub fn values(&self) -> &[T];

    // Constructors, bounded on Float + Send + Sync + AsPrimitive<usize> + Sum + AddAssign + DivAssign + MulAssign
    pub fn from_metric<U, F>(data: &[U], perplexity: T, metric: F) -> Self;
    pub fn from_neighbors(neighbors: &[Vec<Neighbor<T>>], perplexity: T) -> Self;  // now ragged
    pub fn from_edges<I>(n_samples: usize, edges: I) -> Self;
    pub fn from_csr(n_samples, rows, columns, values) -> Result<Self, FromCsrError>;
    pub fn from_csr_unchecked(n_samples, rows, columns, values) -> Self;
    pub fn from_l2(rows: &[&[T]], perplexity: T) -> Self;      // sugar over from_metric with squared-L2
    pub fn from_cosine(rows: &[&[T]], perplexity: T) -> Self;  // sugar over from_metric with chord cosine
}

// Builder for pooling views
pub struct AffinitiesBuilder<'a, T> { ... }
impl<'a, T> AffinitiesBuilder<'a, T> {
    pub fn new(n_samples: usize) -> Self;
    pub fn add(self, weight: T, view: &'a Affinities<T>) -> Self;
    pub fn add_over(self, weight: T, subset: &[usize], view: &'a Affinities<T>) -> Self;  // slice, sort-check
    pub fn add_uniform(self, view: &'a Affinities<T>) -> Self;
    pub fn add_uniform_over(self, subset: &[usize], view: &'a Affinities<T>) -> Self;    // new sugar
    pub fn views_len(&self) -> usize;                                                    // new getter
    pub fn n_samples(&self) -> usize;
    pub fn build(self) -> Affinities<T>;
}

// tSNE type-state builder
pub struct TsneBuilder<'d, T, U, const D: usize = 2> { ... }
impl TsneBuilder {
    pub fn new(data: &'d [U]) -> Self;
    // Config setters (all consume self, return Self)
    pub fn perplexity(self, T) -> Self;
    pub fn epochs(self, usize) -> Self;
    pub fn learning_rate(self, T) -> Self;
    pub fn momentum(self, T) -> Self;
    pub fn final_momentum(self, T) -> Self;
    pub fn momentum_switch_epoch(self, usize) -> Self;
    pub fn stop_lying_epoch(self, usize) -> Self;
    pub fn early_exaggeration(self, T) -> Self;
    pub fn initial_embedding(self, impl Into<Vec<T>>) -> Self;
    pub fn spectral_init(self) -> Self where T: Default, Dim<D>: SpectralBlock;
    pub fn spectral_init_with(self, SpectralParams) -> Self where T: Default, Dim<D>: SpectralBlock;
    pub fn epoch_callback<C: FnMut(usize, &[T]) + 'd>(self, C) -> Self;

    // Affinity setter, only path to the next state
    pub fn with_affinities(self, Affinities<T>) -> TsneReady<'d, T, U, D>;
}

// The affinity-loaded state, no more config setters, only strategy transitions
pub struct TsneReady<'d, T, U, const D: usize> { ... }
impl TsneReady {
    pub fn bhtsne(self, theta: T) -> BhtsneBuilder<'d, T, U, D>;
    pub fn fit_sne(self) -> FitSneBuilder<'d, T, U, D>;
    pub fn exact(self) -> ExactBuilder<'d, T, U, D>;
}

// Per-strategy builders, one per repulsion family
pub struct BhtsneBuilder<'d, T, U, const D: usize> { ... }
impl BhtsneBuilder {
    pub fn fit(self) -> FittedBhtsne<'d, T, U, D>;
    // future: pub fn neural(self, NeuralConfig) -> BhtsneNeuralBuilder;
}

pub struct FitSneBuilder<'d, T, U, const D: usize> { ... }
impl FitSneBuilder {
    pub fn fit(self) -> FittedFitSne<'d, T, U, D>;
    // future: pub fn neural(self, NeuralConfig) -> FitSneNeuralBuilder;
}

pub struct ExactBuilder<'d, T, U, const D: usize> { ... }
impl ExactBuilder {
    pub fn fit(self) -> FittedExact<'d, T, U, D>;
}

// Fitted results, one per strategy
pub struct FittedBhtsne<'d, T, U, const D: usize> { ... }
impl FittedBhtsne {
    pub fn embedding(&self) -> &[T];
    pub fn kl_divergence(&self) -> T;
    pub fn affinities(&self) -> &Affinities<T>;
    // These re-hold the config used, so a caller can inspect what they ran.
}

pub struct FittedFitSne<'d, T, U, const D: usize> { ... }
impl FittedFitSne { same three getters }

pub struct FittedExact<'d, T, U, const D: usize> { ... }
impl FittedExact { same three getters }
```

## Removed from the current tSNE type

- `exact(metric)`, `barnes_hut(theta, metric)`, `barnes_hut_with_neighbors(theta, nbrs)`, `fit_sne(metric)`, `fit_sne_with_neighbors(nbrs)`, `fit_barnes_hut(theta)`, `fit_interpolated()`
- `embedding()`, `kl_divergence()`, `affinities()` on the old tSNE (move to Fitted results)
- The `Fit<T>` enum, `cached_perplexity`, `stop_lying_fired` bookkeeping (moves inside the fit machinery)
- The `#[allow(non_camel_case_types)] struct tSNE` type name itself

## Kept unchanged (internal)

- `bhtsne/src/repulsion.rs` (BarnesHutRepulsion, InterpolatedRepulsion, Repulsion trait)
- `bhtsne/src/tsne/mod.rs` (all internal helpers: search_beta, symmetrize_sparse_matrix, symmetrize_csr, prepare_buffers, clear_buffers, random_init, compute_pairwise_distance_matrix, gradient_descent_step, evaluate_error_*, etc.)
- `bhtsne/src/tsne/vptree.rs`, `bhtsne/src/tsne/spectral.rs`, `bhtsne/src/tsne/interpolation.rs`, `bhtsne/src/tsne/fft.rs`, `bhtsne/src/tsne/morton.rs`
- `bhtsne/src/csv.rs`
- Cargo.toml, feature flags, edition, license, workspace membership

## Module layout

```
bhtsne/src/
├── lib.rs              // crate root: docs + re-exports only
├── error.rs            // FromCsrError, other error types
├── affinities/
│   ├── mod.rs          // Affinities + all constructors + Neighbor
│   └── builder.rs      // AffinitiesBuilder
├── fit/
│   ├── mod.rs          // TsneBuilder + TsneReady + FitConfig internal shared state
│   ├── bhtsne.rs       // BhtsneBuilder + FittedBhtsne
│   ├── fit_sne.rs      // FitSneBuilder + FittedFitSne
│   └── exact.rs        // ExactBuilder + FittedExact
├── repulsion.rs        // unchanged
├── csv.rs              // unchanged
└── tsne/               // unchanged (all internal helpers)
    └── mod.rs
```

## Internal wiring

- `TsneBuilder` and `TsneReady` share a `FitConfig<'d, T, U, D>` struct holding the data reference, all the setter values, the epoch callback, and optional initial embedding / spectral seeder. `TsneReady` adds the injected `Affinities<T>`.
- Each strategy builder holds a `FitConfig` plus a `TsneReady`'s affinities plus strategy-specific fields (theta for BhtsneBuilder, nothing extra for FitSneBuilder and ExactBuilder).
- Each `Fitted*` holds the final embedding, the KL divergence, and the `Affinities<T>` used.

## Fit body

- `BhtsneBuilder::fit` runs `finalize_p_and_seed` then the Barnes-Hut optimisation loop over epochs, exactly the current `run_loop` shape.
- `FitSneBuilder::fit` same shape with the interpolated repulsion strategy. Requires `T: FftNum, Dim<D>: FftDim`.
- `ExactBuilder::fit` runs the dense pairwise loop; unlike the sparse strategies it does NOT consume the injected sparse affinities directly, it densifies internally. See "exact + sparse affinities" note below.

### Exact + sparse affinities

The current `tSNE::exact` builds a dense n**2 P matrix from raw data using the metric. In the new design, `ExactBuilder::fit` receives a sparse `Affinities<T>`. The natural read is to densify: build a dense `Vec<T>` of size `n*n`, fill from the sparse CSR, and run the dense loop. Values not present in the sparse graph are zero. This yields exactly the same numerical fit as the sparse-fed dense algorithm.

Cost: n**2 memory. Fine because exact is already n**2 in time; a caller who chose exact accepted this cost.

## Perplexity handling

- `TsneBuilder::perplexity(p)` sets a config value used only if the caller later configures spectral init or if the fit strategy needs it. Since all strategies consume pre-built affinities, perplexity is no longer used by the fit itself. The setter stays because it belongs to the general t-SNE config (spectral init uses it, epoch callback naming) but has no direct effect on the fit path.

Actually, on reflection: perplexity is unused in the new fit path since Affinities are pre-built. Remove the setter. The perplexity travels with the constructor (`from_metric(data, perplexity, metric)`), not with the fit builder.

## Cache

- Gone. Every fit builds fresh. If the caller wants to reuse a graph, they keep the `Affinities<T>` around and pass it (via `.clone()`) to multiple builders.

## Neural stubs

- `BhtsneBuilder::neural(config)` and `FitSneBuilder::neural(config)` exist as `todo!()` placeholders with a doc comment naming the intended future signature. Not implemented in this PR. They serve to document the extension point.

## Migration guide (users)

Old:
```rust
let mut tsne = tSNE::<f32, &[f32], 2>::new(&samples);
tsne.perplexity(30.0).epochs(1000).barnes_hut(0.5, |a, b| euclidean(a, b));
let emb = tsne.embedding();
```

New:
```rust
let affinities = Affinities::from_metric(&samples, 30.0, |a, b| euclidean(a, b));
let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
    .epochs(1000)
    .with_affinities(affinities)
    .bhtsne(0.5)
    .fit();
let emb = fitted.embedding();
```

Every metric+fit one-shot splits into an Affinities constructor plus a fit chain. Every "with neighbors" one-shot splits into `Affinities::from_neighbors` plus a fit chain. Every "inject affinities" call becomes `with_affinities` on the builder.

## Test migration policy

- Tests that exercised a specific fit method migrate to the equivalent chain.
- Tests that exercised the cache invalidation (e.g. `cached_affinities_invalidated_on_perplexity_change`) delete: the concept is gone.
- Tests that pinned column ordering after fixed-width symmetrise: relax to compare edges and values ignoring column order, since ragged now goes through symmetrize_csr which column-sorts.
- Tests that exercised `barnes_hut_with_neighbors_rejects_ragged_rows`: delete, ragged is now accepted.

## Example migration

- `bhtsne/examples/mnist_pca_tsne.rs`: switch to Affinities + TsneBuilder + BhtsneBuilder chain.
- `bhtsne/examples/fit_vs_bh.rs`: switch to two chains, one bhtsne one fit_sne, compare their embeddings.
- `bhtsne/examples/cora_tsne.rs`, `pubmed_tsne.rs`: `.with_affinities(a).fit_barnes_hut(theta)` becomes `.with_affinities(a).bhtsne(theta).fit()`.

## Experiments migration

- `~/github/multi-affinity-tsne-experiments/src/citation.rs`: swap the `.with_affinities(a).fit_barnes_hut(THETA)` pattern to the new chain, verify warm-cache accuracy identical.
- `~/github/multi-affinity-tsne-experiments/src/sources.rs`: same.
