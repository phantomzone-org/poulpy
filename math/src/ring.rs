pub mod impl_u64;

use crate::dft::DFT;
use crate::modulus::prime::Prime;
use crate::poly::{Poly, PolyRNS};
use crate::modulus::WordOps;
use num::traits::Unsigned;
use std::rc::Rc;

pub struct Ring<O: Unsigned> {
    pub n: usize,
    pub modulus: Prime<O>,
    pub dft: Box<dyn DFT<O>>,
}

impl<O: Unsigned> Ring<O> {

    pub fn log_n(&self) -> usize{
        return self.n().log2();
    }

    pub fn n(&self) -> usize {
        return self.n;
    }

    pub fn new_poly(&self) -> Poly<u64> {
        Poly::<u64>::new(self.n())
    }
}

pub struct RingRNS<O: Unsigned>(pub Vec<Rc<Ring<O>>>);

impl<O: Unsigned> RingRNS<O> {

    pub fn log_n(&self) -> usize{
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
