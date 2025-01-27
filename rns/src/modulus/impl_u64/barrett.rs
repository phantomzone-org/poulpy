use crate::modulus::barrett::{Barrett, BarrettPrecomp};
use crate::modulus::ReduceOnce;
use crate::modulus::{BARRETT, BARRETTLAZY, FOURTIMES, NONE, ONCE, REDUCEMOD, TWICE};

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

impl BarrettPrecomp<u64> {
    pub fn new(q: u64) -> BarrettPrecomp<u64> {
        let big_r: BigUint =
            (BigUint::from(1 as usize) << ((u64::BITS << 1) as usize)) / BigUint::from(q);
        let lo: u64 = (&big_r & BigUint::from(u64::MAX)).to_u64().unwrap();
        let hi: u64 = (big_r >> u64::BITS).to_u64().unwrap();
        let mut precomp: BarrettPrecomp<u64> = Self {
            q: q,
            two_q: q << 1,
            four_q: q << 2,
            lo: lo,
            hi: hi,
            one: Barrett(0, 0),
        };
        precomp.one = precomp.prepare(1);
        precomp
    }

    #[inline(always)]
    pub fn one(&self) -> Barrett<u64> {
        self.one
    }

    #[inline(always)]
    pub fn reduce_assign<const REDUCE: REDUCEMOD>(&self, x: &mut u64) {
        match REDUCE {
            NONE => {}
            ONCE => x.reduce_once_assign(self.q),
            TWICE => x.reduce_once_assign(self.two_q),
            FOURTIMES => x.reduce_once_assign(self.four_q),
            BARRETT => {
                let (_, mhi) = x.widening_mul(self.hi);
                *x = *x - mhi.wrapping_mul(self.q);
                x.reduce_once_assign(self.q);
            }
            BARRETTLAZY => {
                let (_, mhi) = x.widening_mul(self.hi);
                *x = *x - mhi.wrapping_mul(self.q)
            }
            _ => unreachable!("invalid REDUCE argument"),
        }
    }

    #[inline(always)]
    pub fn reduce<const REDUCE: REDUCEMOD>(&self, x: &u64) -> u64 {
        let mut r = *x;
        self.reduce_assign::<REDUCE>(&mut r);
        r
    }

    #[inline(always)]
    pub fn prepare(&self, v: u64) -> Barrett<u64> {
        debug_assert!(v < self.q);
        let quotient: u64 = (((v as u128) << 64) / self.q as u128) as _;
        Barrett(v, quotient)
    }

    #[inline(always)]
    pub fn mul_external<const REDUCE: REDUCEMOD>(&self, lhs: &Barrett<u64>, rhs: &u64) -> u64 {
        let mut r: u64 = *rhs;
        self.mul_external_assign::<REDUCE>(lhs, &mut r);
        r
    }

    #[inline(always)]
    pub fn mul_external_assign<const REDUCE: REDUCEMOD>(&self, lhs: &Barrett<u64>, rhs: &mut u64) {
        let t: u64 = ((*lhs.quotient() as u128 * *rhs as u128) >> 64) as _;
        *rhs = (rhs.wrapping_mul(*lhs.value())).wrapping_sub(self.q.wrapping_mul(t));
        self.reduce_assign::<REDUCE>(rhs);
    }
}
