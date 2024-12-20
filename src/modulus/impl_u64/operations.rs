
use crate::modulus::Operations;
use crate::modulus::prime::Prime;
use crate::modulus::ReduceOnce;
use crate::{apply_unary, apply_binary, apply_ternary};
use itertools::izip;

impl Operations<u64> for Prime<u64>{

    #[inline(always)]
    fn add_binary_assign(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(*b).reduce_once(self.q);
    }

    #[inline(always)]
    fn add_unary_assign(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(*b).reduce_once(self.q);
    }

    #[inline(always)]
    fn add_vec_binary_assign<const CHUNK:usize>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::add_binary_assign, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn add_vec_unary_assign<const CHUNK:usize>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::add_unary_assign, a, b, CHUNK);
    }

    #[inline(always)]
    fn sub_binary_assign(&self, a: &u64, b: &u64, c: &mut u64){
        *c = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn sub_unary_assign(&self, a: &u64, b: &mut u64){
        *b = a.wrapping_add(self.q.wrapping_sub(*b)).reduce_once(self.q);
    }

    #[inline(always)]
    fn sub_vec_binary_assign<const CHUNK:usize>(&self, a: &[u64], b:&[u64], c:&mut [u64]){
        apply_ternary!(self, Self::sub_binary_assign, a, b, c, CHUNK);
    }

    #[inline(always)]
    fn sub_vec_unary_assign<const CHUNK:usize>(&self, a: &[u64], b:&mut [u64]){
        apply_binary!(self, Self::sub_unary_assign, a, b, CHUNK);
    }

    #[inline(always)]
    fn neg_assign(&self, a: &mut u64){
        *a = self.q.wrapping_sub(*a);
    }

    #[inline(always)]
    fn neg_vec_assign<const CHUNK:usize>(&self, a: &mut [u64]){
        apply_unary!(self, Self::neg_assign, a, CHUNK);
    }
}

