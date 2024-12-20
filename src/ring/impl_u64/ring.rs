use crate::ring::Ring;
use crate::dft::ntt::Table;
use crate::modulus::prime::Prime;
use crate::poly::Poly;

impl Ring<u64>{
    pub fn new(n:usize, q_base:u64, q_power:usize) -> Self{
        let prime: Prime<u64> = Prime::<u64>::new(q_base, q_power);
        Self {
            n: n,
            modulus: prime.clone(),
            dft: Box::new(Table::<u64>::new(prime, (2 * n) as u64)),
        }
    }

    pub fn n(&self) -> usize{
        return self.n
    }

    pub fn new_poly(&self) -> Poly<u64>{
        Poly::<u64>::new(self.n())
    }
}