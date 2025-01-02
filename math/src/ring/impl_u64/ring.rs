use crate::ring::Ring;
use crate::dft::ntt::Table;
use crate::modulus::prime::Prime;
use crate::modulus::montgomery::Montgomery;
use crate::poly::Poly;
use crate::modulus::REDUCEMOD;
use crate::modulus::VecOperations;
use crate::CHUNK;

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

impl Ring<u64>{
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

impl Ring<u64>{

    #[inline(always)]
    pub fn add_inplace<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>){
        self.modulus.vec_add_unary_assign::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn add<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &Poly<u64>, c: &mut Poly<u64>){
        self.modulus.vec_add_binary_assign::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn sub_inplace<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>){
        self.modulus.vec_sub_unary_assign::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn sub<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &Poly<u64>, c: &mut Poly<u64>){
        self.modulus.vec_sub_binary_assign::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn neg<const REDUCE: REDUCEMOD>(&self, a: &Poly<u64>, b: &mut Poly<u64>){
        self.modulus.vec_neg_binary_assign::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }

    #[inline(always)]
    pub fn neg_inplace<const REDUCE: REDUCEMOD>(&self, a: &mut Poly<u64>){
        self.modulus.vec_neg_unary_assign::<CHUNK, REDUCE>(&mut a.0);
    }

    #[inline(always)]
    pub fn  mul_montgomery_external<const REDUCE:REDUCEMOD>(&self, a:&Poly<Montgomery<u64>>, b:&Poly<u64>, c: &mut Poly<u64>){
        self.modulus.vec_mul_montgomery_external_binary_assign::<CHUNK, REDUCE>(&a.0, &b.0, &mut c.0);
    }

    #[inline(always)]
    pub fn mul_montgomery_external_inplace<const REDUCE:REDUCEMOD>(&self, a:&Poly<Montgomery<u64>>, b:&mut Poly<u64>){
        self.modulus.vec_mul_montgomery_external_unary_assign::<CHUNK, REDUCE>(&a.0, &mut b.0);
    }
}