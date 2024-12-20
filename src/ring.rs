pub mod impl_u64;

use crate::modulus::prime::Prime;
use crate::dft::DFT;


pub struct Ring<O>{
    pub n:usize,
    pub modulus:Prime<O>,
    pub dft:dyn DFT<O>,
}