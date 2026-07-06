//! Visualize the Cora citation network in 2D with t-SNE under several notions of neighbor, to show
//! how multiple affinity graphs pool into one embedding.
//!
//! Cora is 2708 machine-learning papers (nodes), each with a 1433-dimensional binary bag-of-words
//! feature vector and one of 7 topic labels, connected by 5429 citation edges. The dataset ships in
//! `data/cora.npz` (numpy compressed archive with `features`, `labels`, and `edges` arrays), so the
//! example runs offline.
//!
//! It renders one `cora.png` as a 2x3 panel colored by the 7 topic labels. The top row holds the
//! input representations and the bottom row the naive baseline and the affinity-level fusion:
//!
//! - A, adjacency as a binary vector: each node is the set of its neighbors, compared by cosine.
//! - B, graph shortest-path distances: each node's neighbors are the closest others by hop count.
//! - C, node features: the bag-of-words vectors compared by cosine.
//! - D, naive concatenation: adjacency and features joined into one vector, compared by cosine.
//! - E, mix: the linear pool (union) of the shortest-path graph (B) and the features (C).
//!
//! The graph-specific parts (building the adjacency, the bounded breadth-first search that turns the
//! graph into a neighbor table) live here in the example, not in the crate, which stays a pure t-SNE
//! engine that only ever sees affinity graphs.

use std::{collections::VecDeque, error::Error, time::Instant};

use npyz::npz::NpzArchive;

use plotters::prelude::*;

use bhtsne::{Affinities, AffinitiesBuilder, Neighbor, TsneBuilder};

const N_FEATURES: usize = 1433;
const N_CLASSES: usize = 7;
/// Embedding dimensionality. The Barnes-Hut fit supports 2 or 3.
const DIM: usize = 2;
const PERPLEXITY: f32 = 30.0;
const THETA: f32 = 0.5;
const EPOCHS: usize = 1000;
/// Neighbors per node in the shortest-path view, about `3 * perplexity` as elsewhere in the crate.
const K_HOPS: usize = 90;
/// Topology weight for the fused view: higher follows the graph, lower follows the features.
const ALPHA: f32 = 0.5;
/// Neighbors used by the embedding-quality kNN classifier, the held-out fraction it tests on, and
/// the number of stratified holdout repeats it averages over.
const K_EVAL: usize = 10;
const TEST_FRAC: f32 = 0.3;
const EVAL_REPEATS: usize = 25;

/// Distinct opaque colors for the seven Cora topic classes.
const CLASS_COLORS: [RGBColor; N_CLASSES] = [
    RGBColor(214, 39, 40),
    RGBColor(37, 150, 190),
    RGBColor(26, 152, 69),
    RGBColor(230, 126, 34),
    RGBColor(155, 89, 182),
    RGBColor(22, 160, 133),
    RGBColor(44, 62, 80),
];

const CLASS_NAMES: [&str; N_CLASSES] = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
];

fn main() -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let mut npz = NpzArchive::open("data/cora.npz")?;
    let features: Vec<f32> = npz
        .by_name("features")?
        .expect("missing 'features'")
        .into_vec()?;
    let labels: Vec<u8> = npz
        .by_name("labels")?
        .expect("missing 'labels'")
        .into_vec()?;
    let edges: Vec<u32> = npz.by_name("edges")?.expect("missing 'edges'").into_vec()?;

    let n = labels.len();
    assert_eq!(features.len(), n * N_FEATURES, "feature array is malformed");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Loaded {n} nodes, {} edges, {N_FEATURES} features",
        edges.len() / 2
    );

    // Undirected adjacency, sorted and deduplicated per node.
    let adjacency = build_adjacency(n, &edges);

    // Unreachable nodes in the shortest-path view are placed at twice the graph diameter: far enough
    // to earn negligible affinity, but a real graph distance rather than an arbitrary sentinel.
    let diameter = approx_diameter(&adjacency);
    let pad_distance = 2.0 * diameter as f32;
    println!("Graph diameter estimate {diameter}, padding unreachable nodes at {pad_distance}");

    // The metric-free fit needs only an entity slice of length `n` to fix the sample count, since it
    // never evaluates a metric. Node ids serve as that placeholder.
    let node_ids: Vec<u32> = (0..n as u32).collect();

    let t1 = Instant::now();

    // View A: each node as its adjacency row (a binary vector over nodes), compared by set cosine.
    let adjacency_slices: Vec<&[u32]> = adjacency.iter().map(Vec::as_slice).collect();
    let view_adjacency =
        Affinities::from_metric(&adjacency_slices, PERPLEXITY, |a: &&[u32], b: &&[u32]| {
            set_cosine_distance(a, b)
        });
    println!("  built adjacency-vector affinities");

    // Views B and C are the two fusion inputs, each built directly as an `Affinities`.
    let hop_neighbors: Vec<Vec<Neighbor<f32>>> = (0..n)
        .map(|source| knn_by_hops(&adjacency, source, K_HOPS, pad_distance))
        .collect();
    let view_hops = Affinities::from_neighbors(&hop_neighbors, PERPLEXITY);
    println!("  built shortest-path affinities");

    let normalized = l2_normalize_rows(&features, N_FEATURES);
    let feature_rows: Vec<&[f32]> = normalized.chunks_exact(N_FEATURES).collect();
    let view_features =
        Affinities::from_metric(&feature_rows, PERPLEXITY, |a: &&[f32], b: &&[f32]| {
            feature_cosine_distance(a, b)
        });
    println!("  built feature affinities");

    // View D, the naive baseline: concatenate each node's binary adjacency row with its binary
    // feature row into one vector and embed that single matrix under cosine. Both parts are sets, so
    // the metric indexes the node id and works sparsely rather than materializing the
    // (n + N_FEATURES)-dimensional concatenation. `comb_norm[i]` is the concatenated vector's L2 norm,
    // which uses the feature nonzero count and so assumes the binary bag-of-words this dataset ships.
    let comb_norm: Vec<f32> = (0..n)
        .map(|i| {
            let feat_nnz = features[i * N_FEATURES..(i + 1) * N_FEATURES]
                .iter()
                .filter(|&&v| v != 0.0)
                .count();
            ((adjacency[i].len() + feat_nnz) as f32).sqrt()
        })
        .collect();
    let view_concat = Affinities::from_metric(&node_ids, PERPLEXITY, |a: &u32, b: &u32| {
        let (ia, ib) = (*a as usize, *b as usize);
        let shared_neighbors = sorted_intersection_count(&adjacency[ia], &adjacency[ib]) as f32;
        let fa = &features[ia * N_FEATURES..(ia + 1) * N_FEATURES];
        let fb = &features[ib * N_FEATURES..(ib + 1) * N_FEATURES];
        let shared_words: f32 = fa.iter().zip(fb.iter()).map(|(x, y)| x * y).sum();
        let cosine =
            (shared_neighbors + shared_words) / (comb_norm[ia] * comb_norm[ib] + f32::EPSILON);
        (2.0 * (1.0 - cosine).max(0.0)).sqrt()
    });
    println!("  built naive concatenation affinities");

    // View E fuses the shortest-path graph and the features linearly (union of edges) through the
    // affinities builder.
    let view_mixed = AffinitiesBuilder::new(n)
        .add(ALPHA, &view_hops)
        .add(1.0 - ALPHA, &view_features)
        .build();
    println!("  built mixed (union) affinities");

    // Embed each graph. Every fit is metric-free: it consumes the prebuilt graph and never searches
    // neighbors or evaluates a distance.
    let embed_adjacency = embed(&node_ids, view_adjacency);
    let embed_hops = embed(&node_ids, view_hops);
    let embed_features = embed(&node_ids, view_features);
    let embed_concat = embed(&node_ids, view_concat);
    let embed_mixed = embed(&node_ids, view_mixed);
    println!("  embedded all five views");

    let tsne_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let png_out = "cora.png";
    println!("Writing 2x3 panel to {png_out}...");
    plot_panels(
        &[
            ("A: adjacency vector (cosine)", &embed_adjacency),
            ("B: graph shortest-path hops", &embed_hops),
            ("C: node features (cosine)", &embed_features),
            ("D: naive concat (cosine)", &embed_concat),
            ("E: mix(B, C)", &embed_mixed),
        ],
        &labels,
        png_out,
    )?;
    println!(
        "Done in {:.0} ms (load {load_ms:.0} ms, t-SNE {tsne_ms:.0} ms)",
        load_ms + tsne_ms
    );
    Ok(())
}

/// Builds an undirected adjacency list from a flat `edges` array of `(u, v)` pairs, sorting and
/// deduplicating each node's neighbors.
fn build_adjacency(n: usize, edges: &[u32]) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    for pair in edges.chunks_exact(2) {
        let (u, v) = (pair[0], pair[1]);
        if u != v {
            adjacency[u as usize].push(v);
            adjacency[v as usize].push(u);
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

/// The `k` nearest nodes to `source` by graph hop distance, ascending, excluding `source`. A bounded
/// breadth-first search stops once `k` are collected. If `source`'s connected component holds fewer
/// than `k` other nodes, the row is padded with unreachable nodes at `pad_distance` (twice the graph
/// diameter) so `search_beta` has enough neighbors to converge to the target perplexity. That distance
/// is far enough that the padding gets negligible affinity, yet it is a plausible graph distance
/// rather than an arbitrary sentinel.
fn knn_by_hops(
    adjacency: &[Vec<u32>],
    source: usize,
    k: usize,
    pad_distance: f32,
) -> Vec<Neighbor<f32>> {
    let n = adjacency.len();
    let mut visited = vec![false; n];
    let mut out: Vec<Neighbor<f32>> = Vec::with_capacity(k);
    let mut frontier: VecDeque<(u32, u32)> = VecDeque::new();

    visited[source] = true;
    frontier.push_back((source as u32, 0));

    while let Some((node, hop)) = frontier.pop_front() {
        if node as usize != source {
            out.push(Neighbor {
                index: node as usize,
                distance: hop as f32,
            });
            if out.len() == k {
                break;
            }
        }
        for &next in &adjacency[node as usize] {
            if !visited[next as usize] {
                visited[next as usize] = true;
                frontier.push_back((next, hop + 1));
            }
        }
    }

    // Pad short rows (small components) so search_beta has enough neighbors to converge. `source` is
    // marked visited, so it is never chosen here.
    if out.len() < k {
        for (node, &seen) in visited.iter().enumerate() {
            if out.len() == k {
                break;
            }
            if !seen {
                out.push(Neighbor {
                    index: node,
                    distance: pad_distance,
                });
            }
        }
    }

    out
}

/// The node farthest from `start` by hop distance, with that distance. Nodes in other components are
/// unreachable and ignored.
fn bfs_farthest(adjacency: &[Vec<u32>], start: usize) -> (usize, u32) {
    let mut dist = vec![u32::MAX; adjacency.len()];
    let mut frontier: VecDeque<u32> = VecDeque::new();
    dist[start] = 0;
    frontier.push_back(start as u32);
    let (mut far_node, mut far_dist) = (start, 0u32);

    while let Some(node) = frontier.pop_front() {
        let d = dist[node as usize];
        if d > far_dist {
            far_dist = d;
            far_node = node as usize;
        }
        for &next in &adjacency[node as usize] {
            if dist[next as usize] == u32::MAX {
                dist[next as usize] = d + 1;
                frontier.push_back(next);
            }
        }
    }
    (far_node, far_dist)
}

/// Double-sweep estimate of the graph diameter: sweep from the highest-degree node (to land in the
/// giant component), then from the farthest node it reaches. Exact for trees, a tight lower bound in
/// general, which is all the padding distance needs.
fn approx_diameter(adjacency: &[Vec<u32>]) -> u32 {
    let start = (0..adjacency.len())
        .max_by_key(|&i| adjacency[i].len())
        .unwrap_or(0);
    let (far, _) = bfs_farthest(adjacency, start);
    let (_, diameter) = bfs_farthest(adjacency, far);
    diameter.max(1)
}

/// Number of shared elements of two sorted, deduplicated id lists, by merge.
fn sorted_intersection_count(a: &[u32], b: &[u32]) -> u32 {
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0u32);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    inter
}

/// Cosine distance between two sorted neighbor-id sets in the chord (Euclidean-between-unit-vectors)
/// form, `sqrt(2 (1 - |a ∩ b| / sqrt(|a| |b|)))`. This is a true metric (it is literally the
/// Euclidean distance between the two rows once each is L2-normalized), and its square is
/// `2 (1 - cosine)`, which is the Euclidean squared distance the Gaussian kernel in `search_beta`
/// expects. Two identical directions give distance zero; two orthogonal directions give sqrt(2);
/// two antipodal directions give 2. An empty set is treated as orthogonal to everything (sqrt(2)).
fn set_cosine_distance(a: &[u32], b: &[u32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return std::f32::consts::SQRT_2;
    }
    let inter = sorted_intersection_count(a, b) as f32;
    let cosine = inter / ((a.len() as f32) * (b.len() as f32)).sqrt();
    (2.0 * (1.0 - cosine).max(0.0)).sqrt()
}

/// Cosine distance between two L2-normalized feature rows in the chord form, `sqrt(2 (1 - dot))`.
/// This is the Euclidean distance between the two rows once each is L2-normalized, a proper metric
/// whose square is the Euclidean squared distance the Gaussian kernel expects.
fn feature_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (2.0 * (1.0 - dot).max(0.0)).sqrt()
}

/// Returns a copy of the row-major matrix with each row scaled to unit L2 norm. Zero rows are left
/// as zeros.
fn l2_normalize_rows(data: &[f32], row_len: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    for row in out.chunks_exact_mut(row_len) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in row {
                *x /= norm;
            }
        }
    }
    out
}

/// Fits a `DIM`-dimensional embedding from a prebuilt affinity graph through the metric-free
/// Barnes-Hut fit. The result is row-major with `DIM` coordinates per sample.
fn embed(node_ids: &[u32], affinities: Affinities<f32>) -> Vec<f32> {
    let fitted = TsneBuilder::<f32, u32, DIM>::new(node_ids)
        .epochs(EPOCHS)
        .with_affinities(affinities)
        .bhtsne(THETA)
        .fit();
    fitted.embedding().to_vec()
}

/// A tiny LCG step for reproducible stratified splits.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// In-place Fisher-Yates shuffle driven by the LCG.
fn shuffle(indices: &mut [usize], state: &mut u64) {
    for i in (1..indices.len()).rev() {
        let j = (lcg(state) >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

/// Mean, median, and sample standard deviation of a stratified-holdout kNN classifier run on the
/// embedding, a standard measure of how well a layout separates the classes. Each repeat holds out a
/// class-balanced `test_frac` of the points, predicts every held-out point's label by a majority vote
/// of its `k` nearest training points in the embedding, and records the accuracy. The statistics
/// are taken over `repeats` independent stratified splits.
fn stratified_knn_stats(
    embedding: &[f32],
    labels: &[u8],
    k: usize,
    test_frac: f32,
    repeats: usize,
) -> (f32, f32, f32) {
    let n = labels.len();
    let n_classes = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &l) in labels.iter().enumerate() {
        by_class[l as usize].push(i);
    }

    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut accuracies: Vec<f32> = Vec::with_capacity(repeats);
    let mut scratch: Vec<(f32, u8)> = Vec::with_capacity(n);
    for _ in 0..repeats {
        // Stratified split: hold out `test_frac` of each class.
        let mut is_test = vec![false; n];
        for class in &by_class {
            let mut idx = class.clone();
            shuffle(&mut idx, &mut state);
            let n_test = ((idx.len() as f32) * test_frac).round() as usize;
            for &t in &idx[..n_test] {
                is_test[t] = true;
            }
        }
        let train: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();

        let (mut correct, mut total) = (0usize, 0usize);
        for i in 0..n {
            if !is_test[i] {
                continue;
            }
            total += 1;
            let base_i = DIM * i;
            scratch.clear();
            for &t in &train {
                let base_t = DIM * t;
                let dist2 = (0..DIM)
                    .map(|d| {
                        let delta = embedding[base_t + d] - embedding[base_i + d];
                        delta * delta
                    })
                    .sum();
                scratch.push((dist2, labels[t]));
            }
            let kk = k.min(scratch.len());
            scratch.select_nth_unstable_by(kk - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut votes = vec![0u32; n_classes];
            for &(_, l) in &scratch[..kk] {
                votes[l as usize] += 1;
            }
            let pred = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map(|(c, _)| c as u8)
                .unwrap();
            if pred == labels[i] {
                correct += 1;
            }
        }
        accuracies.push(correct as f32 / total.max(1) as f32);
    }

    let mean = accuracies.iter().sum::<f32>() / repeats as f32;
    let variance =
        accuracies.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / (repeats.max(2) - 1) as f32;
    let std = variance.sqrt();
    accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if repeats % 2 == 1 {
        accuracies[repeats / 2]
    } else {
        (accuracies[repeats / 2 - 1] + accuracies[repeats / 2]) / 2.0
    };
    (mean, median, std)
}

/// Renders the five embeddings as a 2x3 grid of class-colored scatter plots into one image. Each
/// panel is captioned with how its affinity graph was built and with the stratified-holdout kNN
/// classification accuracy of the layout (mean, median, and standard deviation over the repeats). The
/// top row holds the input representations, the bottom row the naive concatenation and the mix
/// fusion, and the sixth cell the class legend.
fn plot_panels(
    panels: &[(&str, &[f32]); 5],
    labels: &[u8],
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (2000, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let cells = root.split_evenly((2, 3));
    for (cell, (title, embedding)) in cells.iter().zip(panels.iter()) {
        let (mean, median, std) =
            stratified_knn_stats(embedding, labels, K_EVAL, TEST_FRAC, EVAL_REPEATS);
        let metric = format!("kNN acc: mean {mean:.3}, median {median:.3}, sd {std:.3}");
        println!("  {title:26}  {metric}");
        draw_panel(cell, title, &metric, embedding, labels)?;
    }
    draw_legend(&cells[5])?;

    root.present()?;
    Ok(())
}

/// Draws one embedding into `area` as a scatter plot of filled, class-colored dots, with a two-line
/// caption (the build `title` and the `metric` line) in a top strip. The caption is drawn manually
/// rather than through `ChartBuilder::caption`, whose tight text box clips the tops of tall glyphs
/// such as the dot of a `j`.
fn draw_panel(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    metric: &str,
    embedding: &[f32],
    labels: &[u8],
) -> Result<(), Box<dyn Error>> {
    let (title_area, plot_area) = area.split_vertically(62);
    title_area.draw(&Text::new(
        title,
        (20, 8),
        ("sans-serif", 22).into_font().color(&BLACK),
    ))?;
    title_area.draw(&Text::new(
        metric,
        (20, 36),
        ("sans-serif", 15).into_font().color(&RGBColor(90, 90, 90)),
    ))?;

    // Bounding box over the two embedding dimensions.
    let mut lo = [f32::INFINITY; 2];
    let mut hi = [f32::NEG_INFINITY; 2];
    for point in embedding.chunks_exact(2) {
        for d in 0..2 {
            lo[d] = lo[d].min(point[d]);
            hi[d] = hi[d].max(point[d]);
        }
    }
    let axis = |d: usize| {
        let pad = (hi[d] - lo[d]).max(f32::EPSILON) * 0.03;
        (lo[d] - pad)..(hi[d] + pad)
    };

    let mut chart = ChartBuilder::on(&plot_area)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(axis(0), axis(1))?;
    chart
        .configure_mesh()
        .light_line_style(TRANSPARENT)
        .draw()?;

    for class in 0..N_CLASSES as u8 {
        let color = CLASS_COLORS[class as usize];
        let points = embedding
            .chunks_exact(2)
            .zip(labels.iter())
            .filter_map(|(c, &l)| (l == class).then_some((c[0], c[1])));
        chart.draw_series(points.map(|pt| Circle::new(pt, 2, color.filled())))?;
    }

    Ok(())
}

/// Draws the class legend as a vertical list of colored swatches and names, filling the grid's
/// eighth cell.
fn draw_legend(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
) -> Result<(), Box<dyn Error>> {
    let (_, height) = area.dim_in_pixel();
    let step = (height as i32 - 120).max(1) / N_CLASSES as i32;
    let x = 60;
    for (i, (name, color)) in CLASS_NAMES.iter().zip(CLASS_COLORS.iter()).enumerate() {
        let y = 90 + i as i32 * step;
        area.draw(&Rectangle::new(
            [(x, y - 12), (x + 24, y + 12)],
            color.filled(),
        ))?;
        area.draw(&Text::new(
            *name,
            (x + 34, y - 9),
            ("sans-serif", 18).into_font().color(&BLACK),
        ))?;
    }
    Ok(())
}
