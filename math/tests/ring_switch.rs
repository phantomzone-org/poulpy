use itertools::izip;
use math::automorphism::AutoPerm;
use math::poly::Poly;
use math::ring::Ring;

#[test]
fn ring_switch_u64() {
    let n: usize = 1 << 4;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring_small: Ring<u64> = Ring::new(n, q_base, q_power);
    let ring_large = Ring::new(2 * n, q_base, q_power);

    sub_test("test_ring_switch_small_to_large_u64::<NTT:false>", || {
        test_ring_switch_small_to_large_u64::<false>(&ring_small, &ring_large)
    });
    sub_test("test_ring_switch_small_to_large_u64::<NTT:true>", || {
        test_ring_switch_small_to_large_u64::<true>(&ring_small, &ring_large)
    });
    sub_test("test_ring_switch_large_to_small_u64::<NTT:false>", || {
        test_ring_switch_large_to_small_u64::<false>(&ring_small, &ring_large)
    });
    sub_test("test_ring_switch_large_to_small_u64::<NTT:true>", || {
        test_ring_switch_large_to_small_u64::<true>(&ring_small, &ring_large)
    });
}

fn sub_test<F: FnOnce()>(name: &str, f: F) {
    println!("Running {}", name);
    f();
}

fn test_ring_switch_small_to_large_u64<const NTT: bool>(
    ring_small: &Ring<u64>,
    ring_large: &Ring<u64>,
) {
    let mut a: Poly<u64> = ring_small.new_poly();
    let mut buf: Poly<u64> = ring_small.new_poly();
    let mut b: Poly<u64> = ring_large.new_poly();

    a.0.iter_mut().enumerate().for_each(|(i, x)| *x = i as u64);

    if NTT {
        ring_small.ntt_inplace::<false>(&mut a);
    }

    ring_large.switch_degree::<NTT>(&a, &mut buf, &mut b);

    if NTT {
        ring_small.intt_inplace::<false>(&mut a);
        ring_large.intt_inplace::<false>(&mut b);
    }

    let gap: usize = ring_large.n() / ring_small.n();

    b.0.iter()
        .step_by(gap)
        .zip(a.0.iter())
        .for_each(|(x_out, x_in)| assert_eq!(x_out, x_in));
}

fn test_ring_switch_large_to_small_u64<const NTT: bool>(
    ring_small: &Ring<u64>,
    ring_large: &Ring<u64>,
) {
    let mut a: Poly<u64> = ring_large.new_poly();
    let mut buf: Poly<u64> = ring_large.new_poly();
    let mut b: Poly<u64> = ring_small.new_poly();

    a.0.iter_mut().enumerate().for_each(|(i, x)| *x = i as u64);

    if NTT {
        ring_large.ntt_inplace::<false>(&mut a);
    }

    ring_large.switch_degree::<NTT>(&a, &mut buf, &mut b);

    if NTT {
        ring_large.intt_inplace::<false>(&mut a);
        ring_small.intt_inplace::<false>(&mut b);
    }

    let gap: usize = ring_large.n() / ring_small.n();

    a.0.iter()
        .step_by(gap)
        .zip(b.0.iter())
        .for_each(|(x_out, x_in)| assert_eq!(x_out, x_in));
}
