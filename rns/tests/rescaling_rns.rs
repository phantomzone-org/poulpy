use itertools::izip;
use num_bigint::BigInt;
use rns::num_bigint::Div;
use rns::poly::{Poly, PolyRNS};
use rns::ring::RingRNS;
use sampling::source::Source;

#[test]
fn rescaling_rns_u64() {
    let n = 1 << 10;
    let moduli: Vec<u64> = vec![
        0x1fffffffffc80001u64,
        0x1fffffffffe00001u64,
        0x1fffffffffb40001,
        0x1fffffffff500001,
    ];
    let ring_rns: RingRNS<u64> = RingRNS::new(n, moduli);

    sub_test("test_div_by_last_modulus::<ROUND:false, NTT:false>", || {
        test_div_by_last_modulus::<false, false>(&ring_rns)
    });
    sub_test("test_div_by_last_modulus::<ROUND:false, NTT:true>", || {
        test_div_by_last_modulus::<false, true>(&ring_rns)
    });
    sub_test("test_div_by_last_modulus::<ROUND:true, NTT:false>", || {
        test_div_by_last_modulus::<true, false>(&ring_rns)
    });
    sub_test("test_div_by_last_modulus::<ROUND:true, NTT:true>", || {
        test_div_by_last_modulus::<true, true>(&ring_rns)
    });
    sub_test(
        "test_div_by_last_modulus_inplace::<ROUND:false, NTT:false>",
        || test_div_by_last_modulus_inplace::<false, false>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_modulus_inplace::<ROUND:false, NTT:true>",
        || test_div_by_last_modulus_inplace::<false, true>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_modulus_inplace::<ROUND:true, NTT:true>",
        || test_div_by_last_modulus_inplace::<true, true>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_modulus_inplace::<ROUND:true, NTT:false>",
        || test_div_by_last_modulus_inplace::<true, false>(&ring_rns),
    );
    sub_test("test_div_by_last_moduli::<ROUND:false, NTT:false>", || {
        test_div_by_last_moduli::<false, false>(&ring_rns)
    });
    sub_test("test_div_by_last_moduli::<ROUND:false, NTT:true>", || {
        test_div_by_last_moduli::<false, true>(&ring_rns)
    });
    sub_test("test_div_by_last_moduli::<ROUND:true, NTT:false>", || {
        test_div_by_last_moduli::<true, false>(&ring_rns)
    });
    sub_test("test_div_by_last_moduli::<ROUND:true, NTT:true>", || {
        test_div_by_last_moduli::<true, true>(&ring_rns)
    });
    sub_test(
        "test_div_by_last_moduli_inplace::<ROUND:false, NTT:false>",
        || test_div_by_last_moduli_inplace::<false, false>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_moduli_inplace::<ROUND:false, NTT:true>",
        || test_div_by_last_moduli_inplace::<false, true>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_moduli_inplace::<ROUND:true, NTT:false>",
        || test_div_by_last_moduli_inplace::<true, false>(&ring_rns),
    );
    sub_test(
        "test_div_by_last_moduli_inplace::<ROUND:true, NTT:true>",
        || test_div_by_last_moduli_inplace::<true, true>(&ring_rns),
    );
}

fn sub_test<F: FnOnce()>(name: &str, f: F) {
    println!("Running {}", name);
    f();
}

fn test_div_by_last_modulus<const ROUND: bool, const NTT: bool>(ring_rns: &RingRNS<u64>) {
    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut buf: [Poly<u64>; 2] = [ring_rns.new_poly(), ring_rns.new_poly()];
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

    ring_rns.div_by_last_modulus::<ROUND, NTT>(&a, &mut buf, &mut c);

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
        if ROUND {
            *a = a.div_round(&scalar_big);
        } else {
            *a = a.div_floor(&scalar_big);
        }
    });

    izip!(coeffs_a, coeffs_c).for_each(|(a, b)| assert_eq!(a, b));
}

fn test_div_by_last_modulus_inplace<const ROUND: bool, const NTT: bool>(ring_rns: &RingRNS<u64>) {
    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut buf: [Poly<u64>; 2] = [ring_rns.new_poly(), ring_rns.new_poly()];

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

    ring_rns.div_by_last_modulus_inplace::<ROUND, NTT>(&mut buf, &mut a);

    if NTT {
        ring_rns
            .at_level(a.level() - 1)
            .intt_inplace::<false>(&mut a);
    }

    // Exports c to coeffs_c
    let mut coeffs_c = vec![BigInt::from(0); a.n()];
    ring_rns
        .at_level(a.level() - 1)
        .to_bigint_inplace(&a, 1, &mut coeffs_c);

    // Performs floor division on a
    let scalar_big = BigInt::from(ring_rns.0[ring_rns.level()].modulus.q);
    coeffs_a.iter_mut().for_each(|a| {
        if ROUND {
            *a = a.div_round(&scalar_big);
        } else {
            *a = a.div_floor(&scalar_big);
        }
    });

    izip!(coeffs_a, coeffs_c).for_each(|(a, b)| assert_eq!(a, b));
}

fn test_div_by_last_moduli<const ROUND: bool, const NTT: bool>(ring_rns: &RingRNS<u64>) {
    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let nb_moduli_dropped: usize = ring_rns.level();

    let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut buf0: [Poly<u64>; 2] = [ring_rns.new_poly(), ring_rns.new_poly()];
    let mut buf1: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut c: PolyRNS<u64> = ring_rns
        .at_level(ring_rns.level() - nb_moduli_dropped)
        .new_polyrns();

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

    ring_rns.div_by_last_moduli::<ROUND, NTT>(nb_moduli_dropped, &a, &mut buf0, &mut buf1, &mut c);

    if NTT {
        ring_rns.at_level(c.level()).intt_inplace::<false>(&mut c);
    }

    // Exports c to coeffs_c
    let mut coeffs_c = vec![BigInt::from(0); a.n()];
    ring_rns
        .at_level(c.level())
        .to_bigint_inplace(&c, 1, &mut coeffs_c);

    // Performs floor division on a
    let mut scalar_big = BigInt::from(1);
    (0..nb_moduli_dropped)
        .for_each(|i| scalar_big *= BigInt::from(ring_rns.0[ring_rns.level() - i].modulus.q));
    coeffs_a.iter_mut().for_each(|a| {
        if ROUND {
            *a = a.div_round(&scalar_big);
        } else {
            *a = a.div_floor(&scalar_big);
        }
    });

    izip!(coeffs_a, coeffs_c).for_each(|(a, b)| assert_eq!(a, b));
}

fn test_div_by_last_moduli_inplace<const ROUND: bool, const NTT: bool>(ring_rns: &RingRNS<u64>) {
    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let nb_moduli_dropped: usize = ring_rns.level();

    let mut a: PolyRNS<u64> = ring_rns.new_polyrns();
    let mut buf0: [Poly<u64>; 2] = [ring_rns.new_poly(), ring_rns.new_poly()];
    let mut buf1: PolyRNS<u64> = ring_rns.new_polyrns();

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

    ring_rns.div_by_last_moduli_inplace::<ROUND, NTT>(
        nb_moduli_dropped,
        &mut buf0,
        &mut buf1,
        &mut a,
    );

    if NTT {
        ring_rns
            .at_level(a.level() - nb_moduli_dropped)
            .intt_inplace::<false>(&mut a);
    }

    // Exports c to coeffs_c
    let mut coeffs_c = vec![BigInt::from(0); a.n()];
    ring_rns
        .at_level(a.level() - nb_moduli_dropped)
        .to_bigint_inplace(&a, 1, &mut coeffs_c);

    // Performs floor division on a
    let mut scalar_big = BigInt::from(1);
    (0..nb_moduli_dropped)
        .for_each(|i| scalar_big *= BigInt::from(ring_rns.0[ring_rns.level() - i].modulus.q));
    coeffs_a.iter_mut().for_each(|a| {
        if ROUND {
            *a = a.div_round(&scalar_big);
        } else {
            *a = a.div_floor(&scalar_big);
        }
    });

    izip!(coeffs_a, coeffs_c).for_each(|(a, b)| assert_eq!(a, b));
}
