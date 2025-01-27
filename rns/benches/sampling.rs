use criterion::{criterion_group, criterion_main, Criterion};
use math::poly::PolyRNS;
use math::ring::RingRNS;
use sampling::source::Source;

fn fill_uniform(c: &mut Criterion) {
    fn runner(r: RingRNS<u64>) -> Box<dyn FnMut()> {
        let mut a: PolyRNS<u64> = r.new_polyrns();
        let seed: [u8; 32] = [0; 32];
        let mut source: Source = Source::new(seed);

        Box::new(move || {
            r.fill_uniform(&mut source, &mut a);
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("fill_uniform");
    for log_n in 11..18 {
        let n = 1 << log_n;
        let moduli: Vec<u64> = vec![
            0x1fffffffffe00001u64,
            0x1fffffffffc80001u64,
            0x1fffffffffb40001,
            0x1fffffffff500001,
        ];

        let ring_rns: RingRNS<u64> = RingRNS::new(n, moduli);

        let runners = [(format!("prime/n={}/level={}", n, ring_rns.level()), {
            runner(ring_rns)
        })];

        for (name, mut runner) in runners {
            b.bench_with_input(name, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, fill_uniform);
criterion_main!(benches);
