pub mod impl_u64;

use crate::modulus::prime::Prime;
use crate::poly::{Poly, PolyRNS};
use num_bigint::BigInt;
use crate::dft::DFT;


pub struct Ring<O>{
    pub n:usize,
    pub modulus:Prime<O>,
    pub dft:Box<dyn DFT<O>>,
}

impl<O> Ring<O>{
    pub fn n(&self) -> usize{
        return self.n
    }

    pub fn new_poly(&self) -> Poly<u64>{
        Poly::<u64>::new(self.n())
    }
}

pub struct RingRNS<'a, O>(& 'a [Ring<O>]);

impl<O: Copy> RingRNS<'_, O> {

    pub fn n(&self) -> usize{
        self.0[0].n()
    }

    pub fn new_polyrns(&self) -> PolyRNS<u64>{
        PolyRNS::<u64>::new(self.n(), self.level())
    }

    pub fn max_level(&self) -> usize{
        self.0.len()-1
    }
    
    pub fn level(&self) -> usize{
        self.0.len()-1
    }

    pub fn at_level(&self, level:usize) -> RingRNS<O>{
        assert!(level <= self.0.len());
        RingRNS(&self.0[..level+1])
    }
}
