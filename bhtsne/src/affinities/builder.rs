//! Builder for pooling several affinity views into a single symmetric [`Affinities`].

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign},
};

use num_traits::{Float, cast::AsPrimitive};

use crate::{affinities::Affinities, tsne};

/// Builder for pooling several affinity views into a single symmetric [`Affinities`].
///
/// [`add`] registers a full-coverage view; [`add_over`] registers a subset-scoped view whose
/// column ids are local and remapped through `subset` at build time. The default pool
/// [`build`] reweights per-view per-row with a softmax of cross-view neighbor agreement; the
/// plain unit-weight pool lives on [`build_uniform`]. All methods return
/// [`Result`](std::result::Result) so caller-supplied invariants surface cleanly.
///
/// [`add`]: AffinitiesBuilder::add
/// [`add_over`]: AffinitiesBuilder::add_over
/// [`build`]: AffinitiesBuilder::build
/// [`build_uniform`]: AffinitiesBuilder::build_uniform
pub struct AffinitiesBuilder<'a, T> {
    n_samples: usize,
    views: Vec<BuilderView<'a, T>>,
    tau: T,
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
    /// Empty builder over `n_samples` global samples. Default `tau` for [`build`] is `0.1`.
    ///
    /// [`build`]: AffinitiesBuilder::build
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            views: Vec::new(),
            tau: T::from(0.1).expect("float type covers 0.1"),
        }
    }

    /// Sets the softmax temperature [`build`] uses. Errors on non-positive `tau`.
    ///
    /// [`build`]: AffinitiesBuilder::build
    pub fn tau(mut self, tau: T) -> Result<Self, crate::error::AffinitiesBuilderError> {
        if tau <= T::zero() {
            return Err(crate::error::AffinitiesBuilderError::NonPositiveTau);
        }
        self.tau = tau;
        Ok(self)
    }

    /// Registers a full-coverage view with scalar weight `weight > 0`.
    pub fn add(
        mut self,
        weight: T,
        view: &'a Affinities<T>,
    ) -> Result<Self, crate::error::AffinitiesBuilderError> {
        if weight <= T::zero() {
            return Err(crate::error::AffinitiesBuilderError::NonPositiveWeight);
        }
        if view.n_samples() != self.n_samples {
            return Err(crate::error::AffinitiesBuilderError::SampleCountMismatch {
                view: view.n_samples(),
                builder: self.n_samples,
            });
        }
        self.views.push(BuilderView {
            weight,
            subset: None,
            view,
        });
        Ok(self)
    }

    /// Registers a subset-scoped view; `subset[k]` is the global id of the view's k-th local
    /// sample. `view.n_samples()` must equal `subset.len()` and the subset must not repeat any
    /// id nor exceed the builder's sample count.
    pub fn add_over(
        mut self,
        weight: T,
        subset: &[usize],
        view: &'a Affinities<T>,
    ) -> Result<Self, crate::error::AffinitiesBuilderError> {
        if weight <= T::zero() {
            return Err(crate::error::AffinitiesBuilderError::NonPositiveWeight);
        }
        if view.n_samples() != subset.len() {
            return Err(crate::error::AffinitiesBuilderError::SubsetSizeMismatch {
                view: view.n_samples(),
                subset: subset.len(),
            });
        }
        let n = self.n_samples;
        for &g in subset {
            if g >= n {
                return Err(crate::error::AffinitiesBuilderError::SubsetIdOutOfRange {
                    id: g,
                    n_samples: n,
                });
            }
        }
        // Check for duplicates via sort-a-copy-and-check-adjacent.
        let mut sorted = subset.to_vec();
        sorted.sort();
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] {
                return Err(crate::error::AffinitiesBuilderError::DuplicateSubsetId {
                    id: sorted[i],
                });
            }
        }
        self.views.push(BuilderView {
            weight,
            subset: Some(subset.to_vec()),
            view,
        });
        Ok(self)
    }

    /// [`add`] with unit weight.
    ///
    /// [`add`]: AffinitiesBuilder::add
    pub fn add_uniform(
        self,
        view: &'a Affinities<T>,
    ) -> Result<Self, crate::error::AffinitiesBuilderError> {
        self.add(T::one(), view)
    }

    /// [`add_over`] with unit weight.
    ///
    /// [`add_over`]: AffinitiesBuilder::add_over
    pub fn add_uniform_over(
        self,
        subset: &[usize],
        view: &'a Affinities<T>,
    ) -> Result<Self, crate::error::AffinitiesBuilderError> {
        self.add_over(T::one(), subset, view)
    }

    /// Pools with plain unit per-row weights, ignoring cross-view agreement. Baseline for
    /// comparison against [`build`].
    ///
    /// [`build`]: AffinitiesBuilder::build
    pub fn build_uniform(self) -> Result<Affinities<T>, crate::error::AffinitiesBuilderError> {
        if self.views.is_empty() {
            return Err(crate::error::AffinitiesBuilderError::NoViews);
        }

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

        Ok(Affinities {
            rows,
            columns,
            values,
        })
    }

    /// Pools with per-row per-view softmax agreement weights.
    ///
    /// For each present view `v` at row `i`, effective weight is `w_v * exp(A_v(i) / tau)` with
    /// `A_v(i) = |N_v(i) ∩ union_{v' != v} N_{v'}(i)| / |N_v(i)|`, `w_v` from [`add`] /
    /// [`add_over`], and `tau` from [`tau`] (default `0.1`). Rows with fewer than three present
    /// views, or where every present view has `A_v(i) = 0`, fall back to [`build_uniform`].
    ///
    /// [`add`]: AffinitiesBuilder::add
    /// [`add_over`]: AffinitiesBuilder::add_over
    /// [`build_uniform`]: AffinitiesBuilder::build_uniform
    /// [`tau`]: AffinitiesBuilder::tau
    pub fn build(self) -> Result<Affinities<T>, crate::error::AffinitiesBuilderError> {
        if self.views.is_empty() {
            return Err(crate::error::AffinitiesBuilderError::NoViews);
        }
        let tau = self.tau;

        let n = self.n_samples;

        // Per-view lookup from global id to local index inside the view. `-1` means the view
        // does not cover the global sample.
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

        // Materialize, once per view per global sample, the sorted set of global neighbor ids of
        // that sample under that view. The rows on `Affinities` are stored sorted by column
        // already, so a subset remap does not need a resort.
        let per_view_global_neighbors: Vec<Vec<Vec<u32>>> = self
            .views
            .iter()
            .enumerate()
            .map(|(m, v)| {
                let map = &global_to_local[m];
                let subset = v.subset.as_deref();
                (0..n)
                    .map(|g| {
                        let local_idx = map[g];
                        if local_idx < 0 {
                            return Vec::new();
                        }
                        let local_idx = local_idx as usize;
                        let (start, end) = (v.view.rows[local_idx], v.view.rows[local_idx + 1]);
                        let mut row = Vec::with_capacity(end - start);
                        for &c in &v.view.columns[start..end] {
                            let g_col = match subset {
                                None => c,
                                Some(sub) => sub[c as usize] as u32,
                            };
                            row.push(g_col);
                        }
                        // Full-coverage rows are sorted; a subset remap of a sorted set is
                        // sorted only if the subset itself is monotone, which we do not require.
                        if subset.is_some() {
                            row.sort_unstable();
                        }
                        row
                    })
                    .collect()
            })
            .collect();

        // A_v(i): fraction of view v's neighbor set at sample i that at least one other view
        // also flagged. Stored as `agreement[view][sample]`, row-major over views.
        let n_views = self.views.len();
        let mut agreement: Vec<Vec<T>> = vec![vec![T::zero(); n]; n_views];
        let mut union_scratch: Vec<u32> = Vec::new();
        for g in 0..n {
            for m in 0..n_views {
                let own = &per_view_global_neighbors[m][g];
                if own.is_empty() {
                    continue;
                }
                union_scratch.clear();
                for (m_other, per_sample) in per_view_global_neighbors.iter().enumerate() {
                    if m_other == m {
                        continue;
                    }
                    union_scratch.extend_from_slice(&per_sample[g]);
                }
                union_scratch.sort_unstable();
                union_scratch.dedup();
                let overlap = own
                    .iter()
                    .filter(|c| union_scratch.binary_search(c).is_ok())
                    .count();
                agreement[m][g] = T::from(overlap).unwrap() / T::from(own.len()).unwrap();
            }
        }

        // Build the pooled conditional in CSR form (variable row width).
        let mut cond_rows: Vec<usize> = Vec::with_capacity(n + 1);
        let mut cond_cols: Vec<u32> = Vec::new();
        let mut cond_vals: Vec<T> = Vec::new();
        cond_rows.push(0);

        let mut scratch: Vec<(u32, T)> = Vec::new();
        // Per row: which views actually have a non-empty row for the current sample, and the
        // effective per-view weight for that row after agreement reweighting and any fallback.
        let mut present: Vec<(usize, usize, T)> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for g in 0..n {
            scratch.clear();
            present.clear();

            for (m, v) in self.views.iter().enumerate() {
                let local_idx = global_to_local[m][g];
                if local_idx < 0 {
                    continue;
                }
                let local_idx = local_idx as usize;
                let (start, end) = (v.view.rows[local_idx], v.view.rows[local_idx + 1]);
                if end > start {
                    present.push((m, local_idx, T::zero()));
                }
            }

            if present.is_empty() {
                cond_rows.push(cond_cols.len());
                continue;
            }

            // Agreement reweighting needs at least three views to distinguish opinions in a
            // meaningful way: with two views the intersection is symmetric so the only
            // asymmetry comes from row-size differences, which is not the signal we want. Fall
            // back to raw view weights when fewer than three views are present at this row.
            let use_agreement = present.len() >= 3;
            let mut total_weight = T::zero();
            if use_agreement {
                // Softmax over per-view agreement scores with temperature `tau`. The largest
                // logit is subtracted for numeric stability so the exponentials stay well below
                // overflow regardless of `tau`.
                let mut max_logit = T::neg_infinity();
                for &(m, _, _) in present.iter() {
                    let logit = agreement[m][g] / tau;
                    if logit > max_logit {
                        max_logit = logit;
                    }
                }
                for entry in present.iter_mut() {
                    let m = entry.0;
                    let logit = agreement[m][g] / tau;
                    let softmax = (logit - max_logit).exp();
                    let eff = self.views[m].weight * softmax;
                    entry.2 = eff;
                    total_weight += eff;
                }
            } else {
                for entry in present.iter_mut() {
                    let base = self.views[entry.0].weight;
                    entry.2 = base;
                    total_weight += base;
                }
            }

            // Fallback: every present view had zero agreement here. Redo weights ignoring
            // agreement so we still emit a well-normalized row rather than dividing by zero.
            if total_weight <= T::zero() {
                total_weight = T::zero();
                for entry in present.iter_mut() {
                    let base = self.views[entry.0].weight;
                    entry.2 = base;
                    total_weight += base;
                }
            }

            let inv_total = total_weight.recip();
            for &(m, local_idx, eff) in &present {
                if eff <= T::zero() {
                    continue;
                }
                let v = &self.views[m];
                let (start, end) = (v.view.rows[local_idx], v.view.rows[local_idx + 1]);
                let row_sum: T = v.view.values[start..end].iter().copied().sum();
                let inv_row_sum = row_sum.recip();
                let a_m = eff * inv_total;
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

        let (rows, columns, mut values) = tsne::symmetrize_csr(&cond_rows, &cond_cols, &cond_vals);
        tsne::normalize_p_values(&mut values, T::one());

        Ok(Affinities {
            rows,
            columns,
            values,
        })
    }
}
