//! Precomputed sparse affinity graphs for t-SNE fitting.
//!
//! An [`Affinities`] value is a symmetric, CSR-layout graph where the values sum to one. Build
//! one from raw data, precomputed neighbors, edges, or an external CSR triple, then inject it
//! into the fit builder. Each constructor lives in its own submodule alongside its tests, keeping
//! this file small and letting readers jump straight to the code they care about.

mod builder;
mod from_csr;
mod from_edges;
mod from_metric;
mod from_neighbors;
mod from_pca;
mod from_shortest_path;

pub use builder::AffinitiesBuilder;

/// A sample's nearest neighbor: its index and the distance (not a similarity) to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neighbor<T> {
    /// Neighbor sample index.
    pub index: usize,
    /// Distance to the neighbor.
    pub distance: T,
}

/// A precomputed sparse, symmetric affinity graph in CSR layout.
///
/// The `rows` array has length `n_samples + 1`, `columns` holds the column indices as `u32`,
/// and `values` holds the non-negative affinity weights. The graph is symmetric and the values
/// sum to one.
#[derive(Clone)]
pub struct Affinities<T> {
    rows: Vec<usize>,
    columns: Vec<u32>,
    values: Vec<T>,
}

impl<T> Affinities<T> {
    /// Number of samples the graph spans.
    pub fn n_samples(&self) -> usize {
        self.rows.len().saturating_sub(1)
    }

    /// CSR row offsets.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// CSR column indices.
    pub fn columns(&self) -> &[u32] {
        &self.columns
    }

    /// Affinity values.
    pub fn values(&self) -> &[T] {
        &self.values
    }
}
