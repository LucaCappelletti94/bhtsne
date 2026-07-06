//! Spectral embedding initialization via `TsneBuilder::spectral_init` across
//! graph sizes and target dimensions. The affinity graph is built once per size
//! outside the timed region, so each benchmark isolates the spectral solve plus
//! a zero-epoch fit shell.
mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use bhtsne::{Affinities, TsneBuilder};

use common::{euclidean, lcg};

const DIM: usize = 128;
const PERPLEXITY: f32 = 30.0;
const SIZES: [usize; 6] = [500, 1000, 2000, 4000, 10000, 20000];

fn bench_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d2");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
        group.throughput(Throughput::Elements((n * 2) as u64));
        group.bench_with_input(BenchmarkId::new("d2", n), &n, |b, _| {
            b.iter(|| {
                let fitted = TsneBuilder::<f32, &[f32], 2>::new(&samples)
                    .spectral_init()
                    .epochs(0)
                    .with_affinities(affinities.clone())
                    .exact()
                    .fit();
                black_box(fitted.embedding().to_vec());
            });
        });
    }
    group.finish();
}

fn bench_d3(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d3");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
        group.throughput(Throughput::Elements((n * 3) as u64));
        group.bench_with_input(BenchmarkId::new("d3", n), &n, |b, _| {
            b.iter(|| {
                let fitted = TsneBuilder::<f32, &[f32], 3>::new(&samples)
                    .spectral_init()
                    .epochs(0)
                    .with_affinities(affinities.clone())
                    .exact()
                    .fit();
                black_box(fitted.embedding().to_vec());
            });
        });
    }
    group.finish();
}

fn bench_d4(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_init_d4");
    for &n in &SIZES {
        let data = lcg(n, DIM, 0xDEAD_BEEF);
        let samples: Vec<&[f32]> = data.chunks(DIM).collect();
        let affinities = Affinities::from_metric(&samples, PERPLEXITY, |a, b| euclidean(a, b));
        group.throughput(Throughput::Elements((n * 4) as u64));
        group.bench_with_input(BenchmarkId::new("d4", n), &n, |b, _| {
            b.iter(|| {
                let fitted = TsneBuilder::<f32, &[f32], 4>::new(&samples)
                    .spectral_init()
                    .epochs(0)
                    .with_affinities(affinities.clone())
                    .exact()
                    .fit();
                black_box(fitted.embedding().to_vec());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_d2, bench_d3, bench_d4);
criterion_main!(benches);
