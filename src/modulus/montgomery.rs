use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::ReduceOnce;

pub struct Montgomery<O>(O);


impl<O> Montgomery<O>{

    #[inline(always)]
    pub fn new(lhs: O) -> Self{
        Self(lhs)
    }

    #[inline(always)]
    pub fn value(&self) -> &O{
        &self.0
    }

    pub fn value_mut(&mut self) -> &mut O{
        &mut self.0
    }
}

pub struct MontgomeryPrecomp<O>{
    q: O,
    q_barrett: BarrettPrecomp<O>,
    q_inv: O,
}

impl MontgomeryPrecomp<u64>{

    #[inline(always)]
    fn new(&self, q: u64) -> MontgomeryPrecomp<u64>{
        let mut r: u64 = 1;
        let mut q_pow = q;
        for _i in 0..63{
            r = r.wrapping_mul(r);
            q_pow = q_pow.wrapping_mul(q_pow)
        }
        Self{ q: q, q_barrett: BarrettPrecomp::new(q, q), q_inv: q_pow}
    }

    #[inline(always)]
    fn prepare(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_assign(lhs, &mut rhs);
        rhs
    }

    fn prepare_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        self.prepare_lazy_assign(lhs, rhs);
        rhs.value_mut().reduce_once_assign(self.q);
    }

    #[inline(always)]
    fn prepare_lazy(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_lazy_assign(lhs, &mut rhs);
        rhs
    }

    fn prepare_lazy_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        let (mhi, _) = lhs.widening_mul(*self.q_barrett.value_lo());
        *rhs = Montgomery((lhs.wrapping_mul(*self.q_barrett.value_hi()).wrapping_add(mhi)).wrapping_mul(self.q).wrapping_neg());
    }

    #[inline(always)]
    fn mul_external(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut r = self.mul_external_lazy(lhs, rhs);
        r.reduce_once_assign(self.q);
        r
    }

    #[inline(always)]
    fn mul_external_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        self.mul_external_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    #[inline(always)]
    fn mul_external_lazy(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut result = rhs;
        self.mul_external_lazy_assign(lhs, &mut result);
        result
    }

    #[inline(always)]
    fn mul_external_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (mhi, mlo) = lhs.value().widening_mul(*rhs);
        let (hhi, _) = self.q.widening_mul(mlo * self.q_inv);
        *rhs = mhi - hhi + self.q
    }

    #[inline(always)]
    fn mul_internal(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external(lhs, *rhs.value()))
    }

    #[inline(always)]
    fn mul_internal_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_assign(lhs, rhs.value_mut());
    }

    #[inline(always)]
    fn mul_internal_lazy(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external_lazy(lhs, *rhs.value()))
    }

    #[inline(always)]
    fn mul_internal_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_lazy_assign(lhs, rhs.value_mut());
    }
}