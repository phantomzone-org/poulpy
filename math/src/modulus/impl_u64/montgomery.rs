
use crate::modulus::ReduceOnce;
use crate::modulus::montgomery::{MontgomeryPrecomp, Montgomery};
use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::{REDUCEMOD, NONE, ONCE, TWICE, FOURTIMES, BARRETT, BARRETTLAZY};
extern crate test;

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
            two_q: q<<1,
            four_q: q<<2,
            barrett: BarrettPrecomp::new(q), 
            q_inv: q_inv,
            one: 0,
            minus_one:0,
        };

        precomp.one = precomp.prepare::<ONCE>(1);
        precomp.minus_one = q-precomp.one;

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

    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - FULL: maps x to x mod q using Barrett reduction.
    #[inline(always)]
    pub fn reduce<const REDUCE:REDUCEMOD>(&self, x: u64) -> u64{
        let mut r: u64 = x;
        self.reduce_assign::<REDUCE>(&mut r);
        r
    }

    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - FULL: maps x to x mod q using Barrett reduction.
    #[inline(always)]
    pub fn reduce_assign<const REDUCE:REDUCEMOD>(&self, x: &mut u64){
        match REDUCE {
            NONE =>{},
            ONCE =>{x.reduce_once_assign(self.q)},
            TWICE=>{x.reduce_once_assign(self.two_q)},
            FOURTIMES =>{x.reduce_once_assign(self.four_q)},
            BARRETT =>{self.barrett.reduce_assign(x)},
            BARRETTLAZY =>{self.barrett.reduce_lazy_assign(x)},
            _ => unreachable!("invalid REDUCE argument")
        }
    }

    /// Returns lhs * 2^64 mod q as a Montgomery<u64>.
    #[inline(always)]
    pub fn prepare<const REDUCE:REDUCEMOD>(&self, lhs: u64) -> Montgomery<u64>{
        let mut rhs: u64 = 0;
        self.prepare_assign::<REDUCE>(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * 2^64 mod q to rhs.
    #[inline(always)]
    pub fn prepare_assign<const REDUCE:REDUCEMOD>(&self, lhs: u64, rhs: &mut Montgomery<u64>){
        let (_, mhi) = lhs.widening_mul(*self.barrett.value_lo());
        *rhs = (lhs.wrapping_mul(*self.barrett.value_hi()).wrapping_add(mhi)).wrapping_mul(self.q).wrapping_neg();
        self.reduce_assign::<REDUCE>(rhs); 
    }

    /// Returns lhs * (2^64)^-1 mod q as a u64.
    #[inline(always)]
    pub fn unprepare<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>) -> u64{
        let mut rhs = 0u64;
        self.unprepare_assign::<REDUCE>(lhs, &mut rhs);
        rhs
    }

    /// Assigns lhs * (2^64)^-1 mod q to rhs.
    #[inline(always)]
    pub fn unprepare_assign<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (_, r) = self.q.widening_mul(lhs.wrapping_mul(self.q_inv));
        *rhs = self.reduce::<REDUCE>(self.q.wrapping_sub(r));        
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q.
    #[inline(always)]
    pub fn mul_external<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>, rhs: u64) -> u64{
        let mut r: u64 = rhs;
        self.mul_external_assign::<REDUCE>(lhs, &mut r);
        r
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    pub fn mul_external_assign<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>, rhs: &mut u64){
        let (mlo, mhi) = lhs.widening_mul(*rhs);
        let (_, hhi) = self.q.widening_mul(mlo.wrapping_mul(self.q_inv));
        *rhs = self.reduce::<REDUCE>(mhi.wrapping_sub(hhi).wrapping_add(self.q));
    }

    /// Returns lhs * rhs * (2^{64})^-1 mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn mul_internal<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        self.mul_external::<REDUCE>(lhs, rhs)
    }

    /// Assigns lhs * rhs * (2^{64})^-1 mod q to rhs.
    #[inline(always)]
    pub fn mul_internal_assign<const REDUCE:REDUCEMOD>(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.mul_external_assign::<REDUCE>(lhs, rhs);
    }

    #[inline(always)]
    pub fn add_internal(&self, lhs: Montgomery<u64>, rhs: Montgomery<u64>) -> Montgomery<u64>{
        self.barrett.reduce(rhs + lhs)
    }

    /// Assigns lhs + rhs to rhs.
    #[inline(always)]
    pub fn add_internal_lazy_assign(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        *rhs += lhs
    }

    /// Assigns lhs + rhs - q if (lhs + rhs) >= q to rhs.
    #[inline(always)]
    pub fn add_internal_reduce_once_assign<const LAZY:bool>(&self, lhs: Montgomery<u64>, rhs: &mut Montgomery<u64>){
        self.add_internal_lazy_assign(lhs, rhs);
        rhs.reduce_once_assign(self.q);
    }

    /// Returns lhs mod q in range [0, 2q-1].
    #[inline(always)]
    pub fn reduce_lazy_assign(&self, lhs: &mut u64){
        self.barrett.reduce_lazy_assign(lhs)
    }

    /// Returns (x^exponent) * 2^64 mod q.
    #[inline(always)]
    pub fn pow(&self, x: Montgomery<u64>, exponent:u64) -> Montgomery<u64>{
        let mut y: Montgomery<u64> = self.one();
        let mut x_mut: Montgomery<u64> = x;
        let mut i: u64 = exponent;
        while i > 0{
            if i & 1 == 1{
                self.mul_internal_assign::<ONCE>(x_mut, &mut y);
            }
            self.mul_internal_assign::<ONCE>(x_mut, &mut x_mut);
            i >>= 1;
        }

        y.reduce_once_assign(self.q);
        y
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
        let y_mont = m_precomp.prepare::<ONCE>(y);
    	assert!(m_precomp.mul_external::<ONCE>(y_mont, x) == (x as u128 * y as u128 % q as u128) as u64);
    }

    #[bench]
    fn bench_mul_external(b: &mut Bencher){
        let q: u64 = 0x1fffffffffe00001;
    	let m_precomp = montgomery::MontgomeryPrecomp::new(q);
        let mut x: u64 = 0x5f876e514845cc8b;
        let y: u64 = 0xad726f98f24a761a;
        let y_mont = m_precomp.prepare::<ONCE>(y);
        b.iter(|| m_precomp.mul_external_assign::<ONCE>(y_mont, &mut x));
    }
}