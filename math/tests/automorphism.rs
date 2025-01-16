use itertools::izip;
use math::automorphism::AutomorphismPermutation;
use math::poly::Poly;
use math::ring::Ring;

#[test]
fn automorphism_u64() {
    let n: usize = 1 << 4;
    let nth_root: usize = n << 1;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_automorphism_native_u64::<NTT:false>", || {
        test_automorphism_native_u64::<false>(&ring, nth_root)
    });
    sub_test("test_automorphism_native_u64::<NTT:true>", || {
        test_automorphism_native_u64::<true>(&ring, nth_root)
    });

    sub_test("test_automorphism_from_perm_u64::<NTT:false>", || {
        test_automorphism_from_perm_u64::<false>(&ring, nth_root)
    });
    sub_test("test_automorphism_from_perm_u64::<NTT:true>", || {
        test_automorphism_from_perm_u64::<true>(&ring, nth_root)
    });
}

fn sub_test<F: FnOnce()>(name: &str, f: F) {
    println!("Running {}", name);
    f();
}

fn test_automorphism_native_u64<const NTT: bool>(ring: &Ring<u64>, nth_root: usize) {
    let n: usize = ring.n();
    let q: u64 = ring.modulus.q;

    let mut p0: Poly<u64> = ring.new_poly();
    let mut p1: Poly<u64> = ring.new_poly();

    for i in 0..p0.n() {
        p0.0[i] = i as u64
    }

    if NTT {
        ring.ntt_inplace::<false>(&mut p0);
    }

    let gal_el: usize = 2 * nth_root - 1;

    ring.a_apply_automorphism_native_into_b::<NTT>(&p0, gal_el, nth_root, &mut p1);

    if NTT {
        ring.intt_inplace::<false>(&mut p1);
    }

    p0.0[0] = 0;
    for i in 1..p0.n() {
        p0.0[i] = q - (n - i) as u64
    }

    izip!(p0.0, p1.0).for_each(|(a, b)| assert_eq!(a, b));
}

fn test_automorphism_from_perm_u64<const NTT: bool>(ring: &Ring<u64>, nth_root: usize) {
    let n: usize = ring.n();
    let q: u64 = ring.modulus.q;

    let mut p0: Poly<u64> = ring.new_poly();
    let mut p1: Poly<u64> = ring.new_poly();

    for i in 0..p0.n() {
        p0.0[i] = i as u64
    }

    if NTT {
        ring.ntt_inplace::<false>(&mut p0);
    }

    let gal_el: usize = 2 * nth_root - 1;

    let auto_perm = AutomorphismPermutation::new::<NTT>(n, gal_el, nth_root);

    ring.a_apply_automorphism_from_perm_into_b::<NTT>(&p0, &auto_perm, &mut p1);

    if NTT {
        ring.intt_inplace::<false>(&mut p1);
    }

    p0.0[0] = 0;
    for i in 1..p0.n() {
        p0.0[i] = q - (n - i) as u64
    }

    izip!(p0.0, p1.0).for_each(|(a, b)| assert_eq!(a, b));
}
