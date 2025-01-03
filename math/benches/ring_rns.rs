use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::ring::{Ring, RingRNS};
use math::ring::impl_u64::ring_rns::new_rings;
use math::poly::PolyRNS;

fn div_floor_by_last_modulus_ntt(c: &mut Criterion) {
    fn runner(r: RingRNS<u64>) -> Box<dyn FnMut() + '_> {
        
        let a: PolyRNS<u64> = r.new_polyrns();
        let mut b: PolyRNS<u64> = r.new_polyrns();
        let mut c: PolyRNS<u64> = r.new_polyrns();

        Box::new(move || {
            r.div_floor_by_last_modulus_ntt(&a, &mut b, &mut c)
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> = c.benchmark_group("div_floor_by_last_modulus_ntt");
    for log_n in 11..18 {

        let n = 1<<log_n;
        let moduli: Vec<u64> = vec![0x1fffffffffe00001u64, 0x1fffffffffc80001u64, 0x1fffffffffb40001, 0x1fffffffff500001];
        let rings: Vec<Ring<u64>> = new_rings(n, moduli);
        let ring_rns: RingRNS<'_, u64> = RingRNS::new(&rings);

        let runners = [
            (format!("prime/n={}/level={}", n, ring_rns.level()), {
                runner(ring_rns)
            }),
        ];

        for (name, mut runner) in runners {
            b.bench_with_input(name, &(), |b, _| b.iter(&mut runner));
        }
    }
}

criterion_group!(benches, div_floor_by_last_modulus_ntt);
criterion_main!(benches);
