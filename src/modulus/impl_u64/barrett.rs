use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::ReduceOnce;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

impl BarrettPrecomp<u64>{
    pub fn new(q: u64) -> BarrettPrecomp<u64> {
        let big_r: BigUint = (BigUint::from(1 as usize)<<((u64::BITS<<1) as usize)) / BigUint::from(q);
        let lo: u64 = (&big_r & BigUint::from(u64::MAX)).to_u64().unwrap();
        let hi: u64 = (big_r >> u64::BITS).to_u64().unwrap();
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

    /// Assigns lhs mod q to lhs.
    #[inline(always)]
    pub fn reduce_assign(&self, lhs: &mut u64){
        self.reduce_lazy_assign(lhs);
        lhs.reduce_once_assign(self.q);
    }

    /// Assigns lhs mod q in range [0, 2q-1] to lhs.
    #[inline(always)]
    pub fn reduce_lazy_assign(&self, lhs: &mut u64){
        let (_, mhi) = lhs.widening_mul(self.hi);
        *lhs = *lhs - mhi.wrapping_mul(self.q)
    }
}