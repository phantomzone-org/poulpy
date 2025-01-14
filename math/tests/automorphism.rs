use itertools::izip;
use math::modulus::WordOps;
use math::poly::Poly;
use math::ring::impl_u64::packing::StreamRepacker;
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
        test_packing_sparse_u64::<false>(&ring, 1)
    });
    sub_test("test_packing_u64::<NTT:true>", || {
        test_packing_sparse_u64::<true>(&ring, 1)
    });
    sub_test("test_packing_sparse_u64::<NTT:false>", || {
        test_packing_sparse_u64::<false>(&ring, 3)
    });
    sub_test("test_packing_sparse_u64::<NTT:true>", || {
        test_packing_sparse_u64::<true>(&ring, 3)
    });
}

fn test_packing_sparse_u64<const NTT: bool>(ring: &Ring<u64>, gap: usize) {
    let n: usize = ring.n();

    let mut result: Vec<Option<&mut Poly<u64>>> = Vec::with_capacity(n);
    result.resize_with(n, || None);

    let mut polys: Vec<Poly<u64>> = vec![ring.new_poly(); (n+gap-1)/gap];

    polys.iter_mut().enumerate().for_each(|(i , poly)|{
        poly.fill(&((1 + i*gap) as u64));
        if NTT {
            ring.ntt_inplace::<false>(poly);
        }
        result[i*gap] = Some(poly);
    });

    ring.pack::<true, NTT>(&mut result, ring.log_n());

    if let Some(poly) = result[0].as_mut() {

        if NTT {
            ring.intt_inplace::<false>(poly);
        }

        poly.0.iter().enumerate().for_each(|(i, x)| {
            if i % gap == 0 {
                assert_eq!(*x, (1+i) as u64)
            } else {
                assert_eq!(*x, 0u64)
            }
        });
    }
}

#[test]
fn packing_streaming_u64() {
    let n: usize = 1 << 5;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_packing_streaming_dense_u64::<NTT:true>", || {
        test_packing_streaming_dense_u64::<true>(&ring)
    });
}

fn test_packing_streaming_dense_u64<const NTT: bool>(ring: &Ring<u64>) {
    let n: usize = ring.n();

    let mut values: Vec<u64> = vec![0; n];
    values
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i + 1) as u64);

    let gap: usize = 3;

    let mut packer = StreamRepacker::new(ring);

    let mut poly: Poly<u64> = ring.new_poly();
    for i in 0..n {
        let i_rev: usize = i.reverse_bits_msb(ring.log_n() as u32);

        if i_rev % gap == 0 {
            poly.fill(&values[i_rev]);
            if NTT {
                ring.ntt_inplace::<false>(&mut poly);
            }
            packer.add::<NTT>(ring, Some(&poly))
        } else {
            packer.add::<NTT>(ring, None)
        }
    }

    packer.flush::<NTT>(ring);

    let result: &mut Poly<u64> = &mut packer.results[0];

    if NTT {
        ring.intt_inplace::<false>(result);
    }

    result.0.iter().enumerate().for_each(|(i, x)| {
        if i % gap == 0 {
            assert_eq!(*x, values[i] as u64)
        } else {
            assert_eq!(*x, 0u64)
        }
    });
}

#[test]
fn trace_u64() {
    let n: usize = 1 << 5;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_trace::<NTT:false>", || test_trace_u64::<false>(&ring));
    sub_test("test_trace::<NTT:true>", || test_trace_u64::<true>(&ring));
}

fn test_trace_u64<const NTT: bool>(ring: &Ring<u64>) {
    let n: usize = ring.n();

    let mut poly: Poly<u64> = ring.new_poly();

    poly.0
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i + 1) as u64);

    if NTT {
        ring.ntt_inplace::<false>(&mut poly);
    }

    let step_start: usize = 2;

    ring.trace_inplace::<NTT>(step_start, &mut poly);

    if NTT {
        ring.intt_inplace::<false>(&mut poly);
    }

    let gap: usize = 1 << (ring.log_n() - step_start);

    poly.0.iter().enumerate().for_each(|(i, x)| {
        if i % gap == 0 {
            assert_eq!(*x, 1 + i as u64)
        } else {
            assert_eq!(*x, 0u64)
        }
    });
}
