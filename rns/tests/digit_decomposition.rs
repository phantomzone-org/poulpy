use itertools::izip;
use rns::modulus::{WordOps, ONCE};
use rns::poly::Poly;
use rns::ring::Ring;
use sampling::source::Source;

#[test]
fn digit_decomposition() {
    let n: usize = 1 << 4;
    let q_base: u64 = 65537u64;
    let q_power: usize = 1usize;
    let ring: Ring<u64> = Ring::new(n, q_base, q_power);

    sub_test("test_unsigned_digit_decomposition", || {
        test_unsigned_digit_decomposition(&ring)
    });

    sub_test("test_signed_digit_decomposition::<BALANCED=false>", || {
        test_signed_digit_decomposition::<false>(&ring)
    });

    sub_test("test_signed_digit_decomposition::<BALANCED=true>", || {
        test_signed_digit_decomposition::<true>(&ring)
    });
}

fn sub_test<F: FnOnce()>(name: &str, f: F) {
    println!("Running {}", name);
    f();
}

fn test_unsigned_digit_decomposition(ring: &Ring<u64>) {
    let mut a: Poly<u64> = ring.new_poly();
    let mut b: Poly<u64> = ring.new_poly();
    let mut c: Poly<u64> = ring.new_poly();

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);
    ring.fill_uniform(&mut source, &mut a);

    let base: usize = 8;
    let log_q: usize = ring.modulus.q.log2();
    let d: usize = ((log_q + base - 1) / base) as _;

    (0..d).for_each(|i| {
        ring.a_ith_digit_unsigned_base_scalar_b_into_c(i, &a, &base, &mut b);
        ring.a_mul_b_scalar_into_a::<ONCE>(&(1 << (i * base)), &mut b);
        ring.a_add_b_into_b::<ONCE>(&b, &mut c);
    });

    izip!(a.0, c.0).for_each(|(a, c)| assert_eq!(a, c));
}

fn test_signed_digit_decomposition<const BALANCED: bool>(ring: &Ring<u64>) {
    let mut a: Poly<u64> = ring.new_poly();
    let mut b: Poly<u64> = ring.new_poly();
    let mut carry: Poly<u64> = ring.new_poly();
    let mut c: Poly<u64> = ring.new_poly();

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);
    ring.fill_uniform(&mut source, &mut a);

    let base: usize = 8;
    let log_q: usize = ring.modulus.q.log2();
    let d: usize = ((log_q + base - 1) / base) as _;

    (0..d).for_each(|i| {
        ring.a_ith_digit_signed_base_scalar_b_into_c::<BALANCED>(i, &a, &base, &mut carry, &mut b);
        ring.a_mul_b_scalar_into_a::<ONCE>(&(1 << (i * base)), &mut b);
        ring.a_add_b_into_b::<ONCE>(&b, &mut c);
    });

    izip!(a.0, c.0).for_each(|(a, c)| assert_eq!(a, c));
}
