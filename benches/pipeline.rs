//! Criterion-бенчи `RuAccent.process_all` на tiny-моделях (ROADMAP этап 6).
//!
//! Требует `RUACCENT_TINY_WORKDIR` (workdir с 4 ONNX-моделями + словарями);
//! без него бенч пропускается (eprintln, без падения). Числа фиксируются в
//! `docs/benchmarks.md` против Python-эталона (`tools/bench_python.py`).

use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruaccent_rs::RuAccent;

const PHRASES: [&str; 4] = [
    "Я плачу за заказ.",
    "Замок был на холме под замком.",
    "А мука в телеге вызывала муку.",
    "Он произнёс кварделямбу и ушёл.",
];

fn workdir() -> Option<PathBuf> {
    std::env::var("RUACCENT_TINY_WORKDIR").ok().map(Into::into)
}

fn corpus_phrases() -> Vec<String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/golden/corpus.txt");
    std::fs::read_to_string(path)
        .expect("golden/corpus.txt")
        .lines()
        .map(str::to_string)
        .filter(|line| !line.is_empty())
        .collect()
}

fn bench_process_all(c: &mut Criterion) {
    let Some(dir) = workdir() else {
        eprintln!("RUACCENT_TINY_WORKDIR not set; skipping process_all benches");
        return;
    };
    let mut accentor = RuAccent::load(&dir).expect("RuAccent::load");

    let mut group = c.benchmark_group("process_all");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    for phrase in PHRASES {
        group.bench_with_input(BenchmarkId::new("phrase", phrase), phrase, |b, p| {
            b.iter(|| black_box(accentor.process_all(black_box(p))))
        });
    }

    let corpus = corpus_phrases();
    group.bench_with_input(BenchmarkId::new("corpus", ""), &corpus, |b, phrases| {
        b.iter(|| {
            for phrase in phrases {
                black_box(accentor.process_all(black_box(phrase)));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_process_all);
criterion_main!(benches);
