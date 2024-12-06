use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::ReduceOnce;

extern crate test;

/// Montgomery is a generic struct storing
/// an element in the Montgomery domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery<O>(pub O);

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
    barrett: BarrettPrecomp<O>,
    q_inv: O,
    one: Montgomery<O>,
    minus_one: Montgomery<O>,
}

/// MontgomeryPrecomp is a set of methods implemented for MontgomeryPrecomp<u64>
/// enabling Montgomery arithmetic over u64 values.
impl MontgomeryPrecomp<u64>{

    /// Returns an new instance of MontgomeryPrecomp<u64>.
    /// This method will fail if gcd(q, 2^64) != 1.
    #[inline(always)]
    pub fn new(q: u64) -> MontgomeryPrecomp<u64>{
        assert!(q & 1 != 0, "Invalid argument: gcd(q={}, radix=2^64) != 1", q);
        let mut q_inv: u64 = 1;
        let mut q_pow = q;
        for _i in 0..63{
            q_inv = q_inv.wrapping_mul(q_pow);
            q_pow = q_pow.wrapping_mul(q_pow);
        }
        let mut precomp = Self{ 
            q: q, 
            barrett: BarrettPrecomp::new(q), 
            q_inv: q_inv,
            one: Montgomery(0),
            minus_one: Montgomery(0),
        };

        precomp.one = precomp.prepare(1);
        precomp.minus_one = Montgomery(q-precomp.one.value());

        precomp
    }

    /// Returns 2^64 mod q as a Montgomery<u64>.
    #[inline(always)]
    pub fn one(&self) -> Montgomery<u64>{
        self.one
    }

    /// Returns (q-1) * 2^64 mod q as a Montgomery<u64>.
    #[inline(always)]
    pub fn minus_one(&self) -> Montgomery<u64>{
        self.minus_one
    }

    /// Returns lhs * 2^64 mod q as a Montgomery<u64>.
    #[inline(always)]
    pub fn prepare(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * 2^64 mod q to rhs.
    #[inline(always)]
    pub fn prepare_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        self.prepare_lazy_assign(lhs, rhs);
        rhs.value_mut().reduce_once_assign(self.q);
    }

    /// Returns lhs * 2^64 mod q in range [0, 2q-1] as a Montgomery<u64>.
    #[inline(always)]
    pub fn prepare_lazy(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs = Montgomery(0);
        self.prepare_lazy_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * 2^64 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    pub fn prepare_lazy_assign(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        let (_, mhi) = lhs.widening_mul(*self.barrett.value_lo());
        *rhs = Montgomery((lhs.wrapping_mul(*self.barrett.value_hi()).wrapping_add(mhi)).wrapping_mul(self.q).wrapping_neg());
    }

    /// Returns lhs * (2^64)^-1 mod q as a u64.
    #[inline(always)]
    pub fn unprepare(&self, lhs: Montgomery<u64>) -> u64{
        let mut rhs = 0u64;
        self.unprepare_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * (2^64)^-1 mod q to rhs.
    #[inline(always)]
    pub fn unprepare_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        self.unprepare_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
        
    }

    /// Returns lhs * (2^64)^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn unprepare_lazy(&self, lhs: Montgomery<u64>) -> u64{
        let mut rhs = 0u64;
        self.unprepare_lazy_assign(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * (2^64)^-1 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    pub fn unprepare_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (_, r) = self.q.widening_mul(lhs.value().wrapping_mul(self.q_inv));
        *rhs = self.q - r
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q.
    #[inline(always)]
    pub fn mul_external(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut r = self.mul_external_lazy(lhs, rhs);
        r.reduce_once_assign(self.q);
        r
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    pub fn mul_external_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        self.mul_external_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn mul_external_lazy(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut result = rhs;
        self.mul_external_lazy_assign(lhs, &mut result);
        result
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    pub fn mul_external_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (mlo, mhi) = lhs.value().widening_mul(*rhs);
        let (_, hhi) = self.q.widening_mul(mlo.wrapping_mul(self.q_inv));
        *rhs = mhi.wrapping_add(self.q - hhi)
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn mul_internal(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external(lhs, *rhs.value()))
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    pub fn mul_internal_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_assign(lhs, rhs.value_mut());
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn mul_internal_lazy(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.mul_external_lazy(lhs, *rhs.value()))
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1] to rhs.
    #[inline(always)]
    pub fn mul_internal_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_lazy_assign(lhs, rhs.value_mut());
    }

    #[inline(always)]
    pub fn add_internal(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        Montgomery(self.barrett.reduce(rhs.value() + lhs.value()))
    }

    /// Assigns lhs + rhs to rhs.
    #[inline(always)]
    pub fn add_internal_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        *rhs.value_mut() += lhs.value()
    }

    /// Assigns lhs + rhs - q if (lhs + rhs) >= q to rhs.
    #[inline(always)]
    pub fn add_internal_reduce_once_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.add_internal_lazy_assign(lhs, rhs);
        rhs.value_mut().reduce_once_assign(self.q);
    }

    /// Returns (x^exponent) * 2^64 mod q.
    #[inline(always)]
    pub fn pow(&self, x: Montgomery<u64>, exponent:u64) -> Montgomery<u64>{
        let mut y: Montgomery<u64> = self.one();
        let mut x_mut: Montgomery<u64> = x;
        let mut i: u64 = exponent;
        while i > 0{
            if i & 1 == 1{
                self.mul_internal_assign(x_mut, &mut y);
            }
            self.mul_internal_assign(x_mut, &mut x_mut);
            i >>= 1;
        }

        y.value_mut().reduce_once_assign(self.q);
        y
    }
}

/// Returns x^exponent mod q.
/// This function internally instantiate a new MontgomeryPrecomp<u64>
/// To be used when called only a few times and if there 
/// is no Prime instantiated with q.
fn pow(x:u64, exponent:u64, q:u64) -> u64{
    let montgomery: MontgomeryPrecomp<u64> = MontgomeryPrecomp::<u64>::new(q);
    let mut y_mont: Montgomery<u64> = montgomery.one();
    let mut x_mont: Montgomery<u64> = montgomery.prepare(x);
    while exponent > 0{
        if exponent & 1 == 1{
            montgomery.mul_internal_assign(x_mont, &mut y_mont);
        }

        montgomery.mul_internal_assign(x_mont, &mut x_mont);
    }
    
    montgomery.unprepare(y_mont)
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