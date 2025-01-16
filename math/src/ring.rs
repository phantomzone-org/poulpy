pub mod impl_u64;
use crate::dft::DFT;
use crate::modulus::prime::Prime;
use crate::modulus::WordOps;
use crate::poly::{Poly, PolyRNS};
use crate::GALOISGENERATOR;
use num::traits::Unsigned;
use std::rc::Rc;

pub struct Ring<O: Unsigned> {
    pub n: usize,
    pub modulus: Prime<O>,
    pub cyclotomic_order: usize,
    pub dft: Box<dyn DFT<O>>,
}

impl<O: Unsigned> Ring<O> {
    pub fn log_n(&self) -> usize {
        return self.n().log2();
    }

    pub fn n(&self) -> usize {
        return self.n;
    }

    pub fn new_poly(&self) -> Poly<u64> {
        Poly::<u64>::new(self.n())
    }

    pub fn cyclotomic_order(&self) -> usize {
        self.cyclotomic_order
    }

    // Returns GALOISGENERATOR^gen_1 * (-1)^gen_2 mod 2^log_nth_root.
    pub fn galois_element(&self, gen_1: usize, gen_2: bool) -> usize {
        let mut gal_el: usize = 1;
        let mut gen_1_pow: usize = GALOISGENERATOR;
        let mut e: usize = gen_1;
        while e > 0 {
            if e & 1 == 1 {
                gal_el = gal_el.wrapping_mul(gen_1_pow);
            }

            gen_1_pow = gen_1_pow.wrapping_mul(gen_1_pow);
            e >>= 1;
        }

        let nth_root = 1 << self.cyclotomic_order;
        gal_el &= nth_root - 1;

        if gen_2 {
            return nth_root - gal_el;
        }
        gal_el
    }
}

pub struct RingRNS<O: Unsigned>(pub Vec<Rc<Ring<O>>>);

impl<O: Unsigned> RingRNS<O> {
    pub fn log_n(&self) -> usize {
        return self.n().log2();
    }

    pub fn n(&self) -> usize {
        self.0[0].n()
    }

    pub fn new_polyrns(&self) -> PolyRNS<u64> {
        PolyRNS::<u64>::new(self.n(), self.level())
    }

    pub fn new_poly(&self) -> Poly<u64> {
        Poly::<u64>::new(self.n())
    }

    pub fn max_level(&self) -> usize {
        self.0.len() - 1
    }

    pub fn level(&self) -> usize {
        self.0.len() - 1
    }

    pub fn at_level(&self, level: usize) -> RingRNS<O> {
        assert!(level <= self.0.len());
        RingRNS(self.0[..level + 1].to_vec())
    }
}
