use crate::modulus::ReduceOnce;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrettPrecomp<O>{
    q: O,
    lo:O,
    hi:O,
}

impl<O> BarrettPrecomp<O>{

    #[inline(always)]
    pub fn value_hi(&self) -> &O{
        &self.hi
    }

    #[inline(always)]
    pub fn value_lo(&self) -> &O{
        &self.lo
    }
}

impl BarrettPrecomp<u64>{
    pub fn new(q: u64) -> BarrettPrecomp<u64> {
        let mut big_r = BigUint::parse_bytes(b"100000000000000000000000000000000", 16).unwrap();
        big_r = big_r / BigUint::from(q);
        let lo = (&big_r & BigUint::from(u64::MAX)).to_u64().unwrap();
        let hi = (big_r >> 64u64).to_u64().unwrap();
        Self{q, lo, hi}
    }

    /// Returns lhs mod q.
    #[inline(always)]
    pub fn reduce(&self, lhs: u64) -> u64{
        let mut r: u64 = self.reduce_lazy(lhs);
        r.reduce_once_assign(self.q);
        r
    }

    /// Returns lhs mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn reduce_lazy(&self, lhs: u64) -> u64{
        let (_, mhi) = lhs.widening_mul(self.hi);
        lhs - mhi.wrapping_mul(self.q)
    }
}