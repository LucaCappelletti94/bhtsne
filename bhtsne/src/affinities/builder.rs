//! Builder for pooling several affinity views into a single symmetric [`Affinities`].

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use crate::{affinities::Affinities, tsne};

/// Builder for pooling several affinity views into a single symmetric [`Affinities`], with support
/// for views that only opine on a subset of the samples.
///
/// A view added through [`add`] must span the builder's global sample count. A view added through
/// [`add_over`] is scoped to a subset of the builder's samples, identified by their global ids, and
/// only its own samples participate in that view. In both cases a sample the view has no opinion on
/// is one whose row inside the view is empty (or, for a subset view, one whose global id is not in
/// the subset).
///
/// [`build`] combines the views row by row: for each global sample, the views that opine on it
/// contribute their row-normalized conditional in proportion to their weight, weights renormalized
/// across the views actually present. The pooled conditional is then symmetrized to a joint
/// affinity `P_ij = (p(j|i) + p(i|j)) / 2` and normalized to sum to one. A sample no view opines on
/// keeps an empty row and drifts under repulsion alone during the fit, which is the honest
/// behavior.
///
/// # Example
///
/// ```
/// use bhtsne::{Affinities, AffinitiesBuilder};
///
/// let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
/// let samples: Vec<&[f32]> = data.chunks(4).collect();
/// let view_a = Affinities::from_metric(&samples, 3.0, |a: &&[f32], b: &&[f32]| {
///     a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
/// });
/// let view_b = Affinities::from_metric(&samples, 3.0, |a: &&[f32], b: &&[f32]| {
///     a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>()
/// });
/// let pooled = AffinitiesBuilder::new(samples.len())
///     .add(0.5, &view_a)
///     .add(0.5, &view_b)
///     .build();
/// assert_eq!(pooled.n_samples(), samples.len());
/// ```
///
/// [`add`]: AffinitiesBuilder::add
/// [`add_over`]: AffinitiesBuilder::add_over
/// [`build`]: AffinitiesBuilder::build
pub struct AffinitiesBuilder<'a, T> {
    n_samples: usize,
    views: Vec<BuilderView<'a, T>>,
}

/// One view registered with an [`AffinitiesBuilder`], either full-coverage (`subset = None`) or
/// scoped to a subset of the builder's samples.
struct BuilderView<'a, T> {
    weight: T,
    /// `None` means the view covers `0..n_samples` in identity order. `Some(sub)` means the view's
    /// k-th local sample is the global sample `sub[k]`.
    subset: Option<Vec<usize>>,
    view: &'a Affinities<T>,
}

// ─── accessors (no trait bound) ───

impl<'a, T> AffinitiesBuilder<'a, T> {
    /// Global sample count the builder pools views over.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of views registered so far.
    pub fn views_len(&self) -> usize {
        self.views.len()
    }
}

// ─── builder methods (standard Float bound) ───

impl<'a, T> AffinitiesBuilder<'a, T>
where
    T: Float + Send + Sync + AsPrimitive<usize> + Sum + AddAssign + DivAssign + MulAssign,
{
    /// Creates an empty builder scoped to `n_samples` global samples.
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            views: Vec::new(),
        }
    }

    /// Adds a full-coverage view weighted by `weight`. The view must have `n_samples()` equal to
    /// the builder's sample count; samples the view has no opinion on are those with empty rows.
    ///
    /// # Panics
    ///
    /// If `weight` is not strictly positive, or if `view.n_samples()` does not equal the builder's.
    pub fn add(mut self, weight: T, view: &'a Affinities<T>) -> Self {
        assert!(
            weight > T::zero(),
            "error: view weight must be strictly positive."
        );
        assert_eq!(
            view.n_samples(),
            self.n_samples,
            "error: view spans {} samples, expected {}.",
            view.n_samples(),
            self.n_samples,
        );
        self.views.push(BuilderView {
            weight,
            subset: None,
            view,
        });
        self
    }

    /// Adds a subset-scoped view weighted by `weight`. `subset[k]` is the global id of the k-th
    /// local sample; `view` must have `n_samples()` equal to `subset.len()`. The view's column
    /// indices are interpreted as local indices into `subset` and remapped to their global ids at
    /// [`build`] time. Rows for global samples outside `subset` are treated as empty in this view.
    ///
    /// # Panics
    ///
    /// If `weight` is not strictly positive, `view.n_samples()` does not equal `subset.len()`, any
    /// subset id is out of range, or `subset` has duplicates.
    ///
    /// [`build`]: AffinitiesBuilder::build
    pub fn add_over(mut self, weight: T, subset: &[usize], view: &'a Affinities<T>) -> Self {
        assert!(
            weight > T::zero(),
            "error: view weight must be strictly positive."
        );
        assert_eq!(
            view.n_samples(),
            subset.len(),
            "error: subset view spans {} samples, but subset has {} ids.",
            view.n_samples(),
            subset.len(),
        );
        let n = self.n_samples;
        for &g in subset {
            assert!(g < n, "error: subset id {g} is out of range 0..{n}.");
        }
        // Check for duplicates via sort-a-copy-and-check-adjacent.
        let mut sorted = subset.to_vec();
        sorted.sort();
        for i in 1..sorted.len() {
            assert!(
                sorted[i] != sorted[i - 1],
                "error: subset has duplicate global id {}.",
                sorted[i]
            );
        }
        self.views.push(BuilderView {
            weight,
            subset: Some(subset.to_vec()),
            view,
        });
        self
    }

    /// Convenience for [`add`] with unit weight, useful when every view has the same weight and the
    /// builder's normalization is doing the work.
    ///
    /// [`add`]: AffinitiesBuilder::add
    pub fn add_uniform(self, view: &'a Affinities<T>) -> Self {
        self.add(T::one(), view)
    }

    /// Convenience for [`add_over`] with unit weight.
    ///
    /// [`add_over`]: AffinitiesBuilder::add_over
    pub fn add_uniform_over(self, subset: &[usize], view: &'a Affinities<T>) -> Self {
        self.add_over(T::one(), subset, view)
    }

    /// Pools the accumulated views into a single symmetric [`Affinities`], as described in the
    /// type-level documentation.
    ///
    /// # Panics
    ///
    /// If no view has been added.
    pub fn build(self) -> Affinities<T> {
        assert!(
            !self.views.is_empty(),
            "error: builder needs at least one view."
        );

        let n = self.n_samples;

        // Per-view lookup from global id to local index inside the view. `-1` means the view does
        // not cover the global sample.
        let global_to_local: Vec<Vec<i32>> = self
            .views
            .iter()
            .map(|v| {
                let mut map = vec![-1i32; n];
                match &v.subset {
                    None => {
                        for (local, slot) in map.iter_mut().enumerate() {
                            *slot = local as i32;
                        }
                    }
                    Some(sub) => {
                        for (local, &g) in sub.iter().enumerate() {
                            map[g] = local as i32;
                        }
                    }
                }
                map
            })
            .collect();

        // Build the pooled conditional in CSR form (variable row width).
        let mut cond_rows: Vec<usize> = Vec::with_capacity(n + 1);
        let mut cond_cols: Vec<u32> = Vec::new();
        let mut cond_vals: Vec<T> = Vec::new();
        cond_rows.push(0);

        // Reused per row: sorted list of `(global column, contribution)` pairs, then folded.
        let mut scratch: Vec<(u32, T)> = Vec::new();
        // Reused per row: which views actually have a non-empty row for the current sample.
        let mut present: Vec<(usize, usize)> = Vec::new();

        // `g` is the global sample id we build a pooled row for, not merely an index; clippy
        // does not see the semantic use because the only textual site is the `global_to_local`
        // lookup, but restructuring around a transposed iterator would be strictly worse.
        #[allow(clippy::needless_range_loop)]
        for g in 0..n {
            scratch.clear();
            present.clear();

            let mut total_weight = T::zero();
            for (m, v) in self.views.iter().enumerate() {
                let local_idx = global_to_local[m][g];
                if local_idx < 0 {
                    continue;
                }
                let local_idx = local_idx as usize;
                let (start, end) = (v.view.rows[local_idx], v.view.rows[local_idx + 1]);
                if end > start {
                    present.push((m, local_idx));
                    total_weight += v.weight;
                }
            }

            if present.is_empty() {
                cond_rows.push(cond_cols.len());
                continue;
            }

            for &(m, local_idx) in &present {
                let v = &self.views[m];
                let (start, end) = (v.view.rows[local_idx], v.view.rows[local_idx + 1]);
                // Recover the row-normalized conditional `p(.|i)` from the (already symmetric)
                // view's row by dividing by the row sum; the joint P is unnormalized enough to
                // matter here, and per-row normalization is what makes presence weighting per-node.
                let row_sum: T = v.view.values[start..end].iter().copied().sum();
                let inv_row_sum = row_sum.recip();
                let a_m = v.weight / total_weight;
                let subset = v.subset.as_deref();
                for k in start..end {
                    let col_local = v.view.columns[k] as usize;
                    let col_global = match subset {
                        None => col_local,
                        Some(sub) => sub[col_local],
                    } as u32;
                    let contribution = a_m * v.view.values[k] * inv_row_sum;
                    scratch.push((col_global, contribution));
                }
            }

            // Fold same-global-column contributions from multiple views into single entries.
            scratch.sort_unstable_by_key(|(c, _)| *c);
            let mut i = 0;
            while i < scratch.len() {
                let c = scratch[i].0;
                let mut acc = T::zero();
                while i < scratch.len() && scratch[i].0 == c {
                    acc += scratch[i].1;
                    i += 1;
                }
                cond_cols.push(c);
                cond_vals.push(acc);
            }
            cond_rows.push(cond_cols.len());
        }

        // Symmetrize the pooled conditional and normalize the joint to sum to one.
        let (rows, columns, mut values) = tsne::symmetrize_csr(&cond_rows, &cond_cols, &cond_vals);
        tsne::normalize_p_values(&mut values, T::one());

        Affinities {
            rows,
            columns,
            values,
        }
    }
}
