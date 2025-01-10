use itertools::izip;
use math::poly::Poly;
use math::ring::impl_u64::ring;
use math::ring::Ring;

#[test]
fn automorphism_u64() {
    let n: usize = 1 << 4;
    let nth_root: usize = n << 1;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_automorphism_u64::<NTT:false>", || {
        test_automorphism_u64::<false>(&ring, nth_root)
    });
    sub_test("test_automorphism_u64::<NTT:true>", || {
        test_automorphism_u64::<true>(&ring, nth_root)
    });
}

fn sub_test<F: FnOnce()>(name: &str, f: F) {
    println!("Running {}", name);
    f();
}

fn test_automorphism_u64<const NTT: bool>(ring: &Ring<u64>, nth_root: usize) {
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

    ring.a_apply_automorphism_into_b::<NTT>(&p0, 2 * n - 1, nth_root, &mut p1);

    if NTT {
        ring.intt_inplace::<false>(&mut p1);
    }

    p0.0[0] = 0;
    for i in 1..p0.n() {
        p0.0[i] = q - (n - i) as u64
    }

    izip!(p0.0, p1.0).for_each(|(a, b)| assert_eq!(a, b));
}

#[test]
fn packing_u64() {
    let n: usize = 1 << 5;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_packing_u64::<NTT:false>", || {
        test_packing_full_u64::<false>(&ring)
    });
    sub_test("test_packing_u64::<NTT:true>", || {
        test_packing_full_u64::<true>(&ring)
    });
    sub_test("test_packing_sparse_u64::<NTT:false>", || {
        test_packing_sparse_u64::<false>(&ring)
    });
    sub_test("test_packing_sparse_u64::<NTT:true>", || {
        test_packing_sparse_u64::<true>(&ring)
    });
}

fn test_packing_full_u64<const NTT: bool>(ring: &Ring<u64>) {
    let n: usize = ring.n();

    let mut result: Vec<Option<Poly<u64>>> = vec![None; n];

    for i in 0..n {
        let mut poly: Poly<u64> = ring.new_poly();
        poly.fill(&(1 + i as u64));
        if NTT {
            ring.ntt_inplace::<false>(&mut poly);
        }

        result[i] = Some(poly);
    }

    ring.pack::<true, NTT>(&mut result, ring.log_n());

    if let Some(poly) = result[0].as_mut() {
        if NTT {
            ring.intt_inplace::<false>(poly);
        }

        poly.0
            .iter()
            .enumerate()
            .for_each(|(i, x)| assert_eq!(*x, 1 + i as u64));
    }
}

fn test_packing_sparse_u64<const NTT: bool>(ring: &Ring<u64>) {
    let n: usize = ring.n();

    let mut result: Vec<Option<Poly<u64>>> = vec![None; n];

    let gap: usize = 3;

    for i in (0..n).step_by(gap) {
        let mut poly: Poly<u64> = ring.new_poly();
        poly.fill(&(1 + i as u64));
        if NTT {
            ring.ntt_inplace::<false>(&mut poly);
        }
        result[i] = Some(poly);
    }

    ring.pack::<true, NTT>(&mut result, ring.log_n());

    if let Some(poly) = result[0].as_mut() {
        if NTT {
            ring.intt_inplace::<false>(poly);
        }

        poly.0.iter().enumerate().for_each(|(i, x)| {
            if i % gap == 0 {
                assert_eq!(*x, 1 + i as u64)
            } else {
                assert_eq!(*x, 0u64)
            }
        });
    }
}
