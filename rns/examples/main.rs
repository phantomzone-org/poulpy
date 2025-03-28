use rns::dft::ntt::Table;
use rns::modulus::prime::Prime;
use rns::ring::Ring;

fn main() {
    // Example usage of `Prime<u64>`
    let q_base: u64 = 65537; // Example prime base
    let q_power: usize = 1; // Example power
    let prime_instance: Prime<u64> = Prime::<u64>::new(q_base, q_power);

    // Display the fields of `Prime` to verify
    println!("Prime instance created:");
    println!("q: {}", prime_instance.q());
    println!("q_base: {}", prime_instance.q_base());
    println!("q_power: {}", prime_instance.q_power());

    let n: usize = 32;
    let nth_root: usize = n << 1;

    let ntt_table: Table<u64> = Table::<u64>::new(prime_instance, nth_root);

    let mut a: Vec<u64> = vec![0; (nth_root >> 1) as usize];

    for i in 0..a.len() {
        a[i] = i as u64;
    }

    println!("{:?}", a);

    ntt_table.forward_inplace::<false>(&mut a);

    println!("{:?}", a);

    ntt_table.backward_inplace::<false>(&mut a);

    println!("{:?}", a);

    let r: Ring<u64> = Ring::<u64>::new(n as usize, q_base, q_power);

    let mut p0: rns::poly::Poly<u64> = r.new_poly();
    let mut p1: rns::poly::Poly<u64> = r.new_poly();

    for i in 0..p0.n() {
        p0.0[i] = i as u64
    }

    r.a_apply_automorphism_native_into_b::<false>(&p0, 2 * r.n - 1, nth_root, &mut p1);

    println!("{:?}", p1);
}
