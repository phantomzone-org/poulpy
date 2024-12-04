use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrettPrecomp<O>(O, O);

impl<O> BarrettPrecomp<O>{

    #[inline(always)]
    pub fn value_hi(&self) -> &O{
        &self.1
    }

    #[inline(always)]
    pub fn value_lo(&self) -> &O{
        &self.0
    }
}

impl BarrettPrecomp<u64>{
    pub fn new(q: u64) -> BarrettPrecomp<u64> {
        let mut big_r = BigUint::parse_bytes(b"100000000000000000000000000000000", 16).unwrap();
        big_r = big_r / BigUint::from(q);
        let lo = (&big_r & BigUint::from(u64::MAX)).to_u64().unwrap();
        let hi = (big_r >> 64u64).to_u64().unwrap();
        Self(lo, hi)
    }
}