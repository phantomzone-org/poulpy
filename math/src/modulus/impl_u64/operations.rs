
use crate::modulus::{WordOperations, VecOperations};
use crate::modulus::prime::Prime;
use crate::modulus::ReduceOnce;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::REDUCEMOD;
use crate::{apply_unary, apply_binary, apply_ternary};
use itertools::izip;

impl WordOperations<u64> for Prime<u64>{

    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - TWO: subtracts 2q if x >= 2q.
    /// - FOUR: subtracts 4q if x >= 4q.
    /// - BARRETT: maps x to x mod q using Barrett reduction.
    /// - BARRETTLAZY: maps x to x mod q using Barrett reduction with values in [0, 2q-1].
    #[inline(always)]
    fn word_reduce_assign<const REDUCE:REDUCEMOD>(&self, x: &mut u64){
        self.montgomery.reduce_assign::<REDUCE>(x);
    }

    #[inline(always)]
    fn word_add_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(*b);
        self.word_reduce_assign::<REDUCE>(c);
    }

    #[inline(always)]
    fn word_add_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(*b);
        self.word_reduce_assign::<REDUCE>(b);
    }

    #[inline(always)]
    fn word_sub_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn word_sub_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn word_neg_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &mut u64){
        *a = self.q.wrapping_sub(*a);
        self.word_reduce_assign::<REDUCE>(a)
    }

    #[inline(always)]
    fn word_neg_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut u64){
        *b = self.q.wrapping_sub(*a);
        self.word_reduce_assign::<REDUCE>(b)
    }

    #[inline(always)]
    fn word_prepare_montgomery_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut Montgomery<u64>){
        self.montgomery.prepare_assign::<REDUCE>(*a, b);
    }

    #[inline(always)]
    fn word_mul_montgomery_external_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &Montgomery<u64>, b:&u64, c: &mut u64){
        *c = self.montgomery.mul_external::<REDUCE>(*a, *b);
    }

    #[inline(always)]
    fn word_mul_montgomery_external_unary_assign<const REDUCE:REDUCEMOD>(&self, lhs:&Montgomery<u64>, rhs:&mut u64){
        self.montgomery.mul_external_assign::<REDUCE>(*lhs, rhs);
    }
}

impl VecOperations<u64> for Prime<u64>{

    /// Applies a modular reduction on x based on REDUCE:
    /// - LAZY: no modular reduction.
    /// - ONCE: subtracts q if x >= q.
    /// - TWO: subtracts 2q if x >= 2q.
    /// - FOUR: subtracts 4q if x >= 4q.
    /// - BARRETT: maps x to x mod q using Barrett reduction.
    /// - BARRETTLAZY: maps x to x mod q using Barrett reduction with values in [0, 2q-1].
    #[inline(always)]
    fn vec_reduce_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, x: &mut [u64]){
        apply_unary!(self, Self::word_reduce_assign::<REDUCE>, x, CHUNK);
    }

    #[inline(always)]
    fn vec_add_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::word_add_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn vec_add_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::word_add_unary_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn vec_sub_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::word_sub_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn vec_sub_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::word_sub_unary_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn vec_neg_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &mut [u64]){
        apply_unary!(self, Self::word_neg_unary_assign::<REDUCE>, a, CHUNK);
    }

    #[inline(always)]
    fn vec_neg_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b: &mut [u64]){
        apply_binary!(self, Self::word_neg_binary_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn vec_prepare_montgomery_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b: &mut [Montgomery<u64>]){
        apply_binary!(self, Self::word_prepare_montgomery_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn vec_mul_montgomery_external_binary_assign<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:& [Montgomery<u64>], b:&[u64], c: &mut [u64]){
        apply_ternary!(self, Self::word_mul_montgomery_external_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn vec_mul_montgomery_external_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:& [Montgomery<u64>], b:&mut [u64]){
        apply_binary!(self, Self::word_mul_montgomery_external_unary_assign::<REDUCE>, a, b, CHUNK);
    }
}
