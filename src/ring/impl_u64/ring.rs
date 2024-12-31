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

    pub fn ntt_inplace<const LAZY:bool>(&self, poly: &mut Poly<u64>){
        match LAZY{
            true => self.dft.forward_inplace_lazy(&mut poly.0),
            false => self.dft.forward_inplace(&mut poly.0)
        }
    }

    pub fn intt_inplace<const LAZY:bool>(&self, poly: &mut Poly<u64>){
        match LAZY{
            true => self.dft.forward_inplace_lazy(&mut poly.0),
            false => self.dft.forward_inplace(&mut poly.0)
        }
    }

    pub fn ntt<const LAZY:bool>(&self, poly_in: &Poly<u64>, poly_out: &mut Poly<u64>){
        poly_out.0.copy_from_slice(&poly_in.0);
        match LAZY{
            true => self.dft.backward_inplace_lazy(&mut poly_out.0),
            false => self.dft.backward_inplace(&mut poly_out.0)
        }
    }

    pub fn intt<const LAZY:bool>(&self, poly_in: &Poly<u64>, poly_out: &mut Poly<u64>){
        poly_out.0.copy_from_slice(&poly_in.0);
        match LAZY{
            true => self.dft.backward_inplace_lazy(&mut poly_out.0),
            false => self.dft.backward_inplace(&mut poly_out.0)
        }
    }
}