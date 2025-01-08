use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::poly::PolyRNS;
use math::ring::RingRNS;

fn div_floor_by_last_modulus_ntt_true(c: &mut Criterion) {
    fn runner(r: RingRNS<u64>) -> Box<dyn FnMut()> {
        let a: PolyRNS<u64> = r.new_polyrns();
        let mut b: [math::poly::Poly<u64>; 2] = [r.new_poly(), r.new_poly()];
        let mut c: PolyRNS<u64> = r.new_polyrns();

        Box::new(move || r.div_by_last_modulus::<false, true>(&a, &mut b, &mut c))
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("div_floor_by_last_modulus_ntt_true");
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

criterion_group!(benches, div_floor_by_last_modulus_ntt_true);
criterion_main!(benches);
