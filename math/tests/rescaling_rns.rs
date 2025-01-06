use math::poly::PolyRNS;
use math::ring::RingRNS;
use num_bigint::BigInt;
use num_bigint::Sign;
use sampling::source::Source;

#[test]
fn rescaling_rns_u64() {
    let n = 1 << 10;
    let moduli: Vec<u64> = vec![0x1fffffffffc80001u64, 0x1fffffffffe00001u64];
    let ring_rns: RingRNS<u64> = RingRNS::new(n, moduli);

    test_div_floor_by_last_modulus::<false>(&ring_rns);
    test_div_floor_by_last_modulus::<true>(&ring_rns);
}

fn test_div_floor_by_last_modulus<const NTT: bool>(ring_rns: &RingRNS<u64>) {
    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut b: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut c: PolyRNS<u64> = ring_rns.at_level(ring_rns.level() - 1).new_polyrns();

    // Allocates a random PolyRNS
    ring_rns.fill_uniform(&mut source, &mut a);

    // Maps PolyRNS to [BigInt]
    let mut coeffs_a: Vec<BigInt> = (0..a.n()).map(|i| BigInt::from(i)).collect();
    ring_rns
        .at_level(a.level())
        .to_bigint_inplace(&a, 1, &mut coeffs_a);

    // Performs c = intt(ntt(a) / q_level)
    if NTT {
        ring_rns.ntt_inplace::<false>(&mut a);
    }

    ring_rns.div_floor_by_last_modulus::<NTT>(&a, &mut b, &mut c);

    if NTT {
        ring_rns.at_level(c.level()).intt_inplace::<false>(&mut c);
    }

    // Exports c to coeffs_c
    let mut coeffs_c = vec![BigInt::from(0); c.n()];
    ring_rns
        .at_level(c.level())
        .to_bigint_inplace(&c, 1, &mut coeffs_c);

    // Performs floor division on a
    let scalar_big = BigInt::from(ring_rns.0[ring_rns.level()].modulus.q);
    coeffs_a.iter_mut().for_each(|a| {
        // Emulates floor division in [0, q-1] and maps to [-(q-1)/2, (q-1)/2-1]
        *a /= &scalar_big;
        if a.sign() == Sign::Minus {
            *a -= 1;
        }
    });

    assert!(coeffs_a == coeffs_c);
}
