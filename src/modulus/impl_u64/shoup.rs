use crate::modulus::ReduceOnce;
use crate::modulus::shoup::{ShoupPrecomp, Shoup};

impl ShoupPrecomp<u64>{

    pub fn new(q: u64) -> Self {
        let mut precomp: ShoupPrecomp<u64> = Self{q:q, one:Shoup(0,0)};
        precomp.one = precomp.prepare(1);
        precomp
    }

    #[inline(always)]
    pub fn one(&self) -> Shoup<u64> {
        self.one
    }

    #[inline(always)]
    pub fn prepare(&self, v: u64) -> Shoup<u64> {
        debug_assert!(v < self.q);
        let quotient: u64 = (((v as u128) << 64) / self.q as u128) as _;
        Shoup(v, quotient)
    }

    #[inline(always)]
    pub fn mul_external(&self, lhs: Shoup<u64>, rhs: u64) -> u64 {
        let mut r: u64 = self.mul_external_lazy(lhs, rhs);
        r.reduce_once_assign(self.q);
        r
    }

    #[inline(always)]
    pub fn mul_external_assign(&self, lhs: Shoup<u64>, rhs: &mut u64){
        self.mul_external_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    #[inline(always)]
    pub fn mul_external_lazy(&self, lhs: Shoup<u64>, rhs: u64) -> u64 {
        let mut r: u64 = rhs;
        self.mul_external_lazy_assign(lhs, &mut r);
        r
    }

    #[inline(always)]
    pub fn mul_external_lazy_assign(&self, lhs: Shoup<u64>, rhs: &mut u64){
        let t: u64 = ((*lhs.quotient() as u128 * *rhs as u128) >> 64) as _;
        *rhs = (rhs.wrapping_mul(*lhs.value())).wrapping_sub(self.q.wrapping_mul(t));
    }

    #[inline(always)]
    pub fn reduce_assign(&self, rhs: &mut u64){
        self.reduce_assign_lazy(rhs);
        rhs.reduce_once_assign(self.q);
    }

    #[inline(always)]
    pub fn reduce_assign_lazy(&self, rhs: &mut u64){
        *rhs = rhs.wrapping_sub(self.q.wrapping_mul(((self.one.1 as u128 * *rhs as u128) >> 64) as _))
    }
}