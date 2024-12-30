
use crate::modulus::Operations;
use crate::modulus::prime::Prime;
use crate::modulus::ReduceOnce;
use crate::modulus::montgomery::Montgomery;
use crate::modulus::{REDUCEMOD, NONE, ONCE, BARRETT, BARRETTLAZY};
use crate::{apply_unary, apply_binary, apply_ternary};
use itertools::izip;

impl Operations<u64> for Prime<u64>{

    #[inline(always)]
    fn add_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(*b);
        self.montgomery.reduce_assign::<REDUCE>(c);
    }

    #[inline(always)]
    fn add_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(*b);
        self.montgomery.reduce_assign::<REDUCE>(b);
    }

    #[inline(always)]
    fn add_vec_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::add_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn add_vec_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::add_unary_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn sub_binary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn sub_unary_assign<const REDUCE:REDUCEMOD>(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn sub_vec_binary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::sub_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn sub_vec_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::sub_unary_assign::<REDUCE>, a, b, CHUNK);
    }

    #[inline(always)]
    fn neg_assign<const REDUCE:REDUCEMOD>(&self, a: &mut u64){
        *a = self.q.wrapping_sub(*a);
        self.montgomery.reduce_assign::<REDUCE>(a)
    }

    #[inline(always)]
    fn neg_vec_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a: &mut [u64]){
        apply_unary!(self, Self::neg_assign::<REDUCE>, a, CHUNK);
    }

    #[inline(always)]
    fn mul_montgomery_external_binary_assign<const REDUCE:REDUCEMOD>(&self, a:& Montgomery<u64>, b:&u64, c: &mut u64){
        *c = self.montgomery.mul_external::<REDUCE>(*a, *b);
    }

    #[inline(always)]
    fn mul_montgomery_external_unary_assign<const REDUCE:REDUCEMOD>(&self, lhs:&Montgomery<u64>, rhs:&mut u64){
        *rhs = self.montgomery.mul_external::<REDUCE>(*lhs, *rhs);
    }

    #[inline(always)]
    fn mul_vec_montgomery_external_binary_assign<const CHUNK:usize,const REDUCE:REDUCEMOD>(&self, a:& [Montgomery<u64>], b:&[u64], c: &mut [u64]){
        apply_ternary!(self, Self::mul_montgomery_external_binary_assign::<REDUCE>, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn mul_vec_montgomery_external_unary_assign<const CHUNK:usize, const REDUCE:REDUCEMOD>(&self, a:&[Montgomery<u64>], b:&mut [u64]){
        apply_binary!(self, Self::mul_montgomery_external_unary_assign::<REDUCE>, a, b, CHUNK);
    }
}
