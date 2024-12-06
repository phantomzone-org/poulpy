extern crate math;
use math::modulus::prime::Prime;
use math::dft::ntt::Table;

fn main() {
    // Example usage of `Prime<u64>`
    let q_base: u64 = 65537;      // Example prime base
    let q_power: u64 = 2;     // Example power
    let mut prime_instance: Prime<u64> = Prime::<u64>::new(q_base, q_power);
    
    // Display the fields of `Prime` to verify
    println!("Prime instance created:");
    println!("q: {}", prime_instance.q());
    println!("q_base: {}", prime_instance.q_base());
    println!("q_power: {}", prime_instance.q_power());

    let n: u64 = 1024;
    let nth_root: u64 = n<<1;

    let ntt_table: Table<'_, u64> = Table::<u64>::new(&mut prime_instance, nth_root);

}