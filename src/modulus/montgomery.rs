use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::ReduceOnce;

extern crate test;

/// Montgomery is a generic struct storing
/// an element in the Montgomery domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery<O>(O);

/// Implements helper methods on the struct Montgomery<O>.
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

/// MontgomeryPrecomp is a generic struct storing 
/// precomputations for Montgomery arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MontgomeryPrecomp<O>{
    q: O,
    q_barrett: BarrettPrecomp<O>,
    q_inv: O,
}

/// MontgomeryPrecomp is a set of methods implemented for MontgomeryPrecomp<u64>
/// enabling Montgomery arithmetic over u64 values.
impl MontgomeryPrecomp<u64>{

    /// Returns an new instance of MontgomeryPrecomp<u64>.
    /// This method will fail if gcd(q, 2^64) != 1.
    #[inline(always)]
    fn new(q: u64) -> MontgomeryPrecomp<u64>{
        assert!(q & 1 != 0, "Invalid argument: gcd(q={}, radix=2^64) != 1", q);
        let mut q_inv: u64 = 1;
        let mut q_pow = q;
        for _i in 0..63{
            q_inv = q_inv.wrapping_mul(q_pow);
            q_pow = q_pow.wrapping_mul(q_pow);
        }
        Self{ q: q, q_barrett: BarrettPrecomp::new(q), q_inv: q_inv}
    }

    /// Returns (lhs<<64)%q as a Montgomery<u64>.
    #[inline(always)]
    fn prepare(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns (lhs<<64)%q to rhs.
    #[inline(always)]
    fn prepare_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        self.prepare_lazy_assign(lhs, rhs);
        rhs.value_mut().reduce_once_assign(self.q);
    }

    /// Returns (lhs<<64)%q in range [0, 2q-1] as a Montgomery<u64>.
    #[inline(always)]
    fn prepare_lazy(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_lazy_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns (lhs<<64)%q in range [0, 2q-1] to rhs.
    #[inline(always)]
    fn prepare_lazy_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        let (_, mhi) = lhs.widening_mul(*self.q_barrett.value_lo());
        *rhs = Montgomery((lhs.wrapping_mul(*self.q_barrett.value_hi()).wrapping_add(mhi)).wrapping_mul(self.q).wrapping_neg());
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q.
    #[inline(always)]
    fn mul_external(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut r = self.mul_external_lazy(lhs, rhs);
        r.reduce_once_assign(self.q);
        r
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    fn mul_external_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        self.mul_external_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    fn mul_external_lazy(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut result = rhs;
        self.mul_external_lazy_assign(lhs, &mut result);
        result
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    fn mul_external_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (mlo, mhi) = lhs.value().widening_mul(*rhs);
        let (_, hhi) = self.q.widening_mul(mlo.wrapping_mul(self.q_inv));
        *rhs = mhi.wrapping_add(self.q - hhi)
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    fn mul_internal(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external(lhs, *rhs.value()))
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    fn mul_internal_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_assign(lhs, rhs.value_mut());
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    fn mul_internal_lazy(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external_lazy(lhs, *rhs.value()))
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    fn mul_internal_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_lazy_assign(lhs, rhs.value_mut());
    }
}


#[cfg(test)]
mod tests {
    use crate::modulus::montgomery;
    use super::*;
    use test::Bencher;

    #[test]
    fn test_mul_external() {
        let q: u64 = 0x1fffffffffe00001;
    	let m_precomp = montgomery::MontgomeryPrecomp::new(q);
        let x: u64 = 0x5f876e514845cc8b;
        let y: u64 = 0xad726f98f24a761a;
        let y_mont = m_precomp.prepare(y);
    	assert!(m_precomp.mul_external(y_mont, x) == (x as u128 * y as u128 % q as u128) as u64);
    }

    #[bench]
    fn bench_mul_external(b: &mut Bencher){
        let q: u64 = 0x1fffffffffe00001;
    	let m_precomp = montgomery::MontgomeryPrecomp::new(q);
        let mut x: u64 = 0x5f876e514845cc8b;
        let y: u64 = 0xad726f98f24a761a;
        let y_mont = m_precomp.prepare(y);
        b.iter(|| m_precomp.mul_external_assign(y_mont, &mut x));
    }
}