use rand_distr::Distribution;
use sampling::source::Source;

use crate::hal::{
    api::{
        VecZnxBigAdd, VecZnxBigAddDistF64, VecZnxBigAddInplace, VecZnxBigAddNormal, VecZnxBigAddSmall, VecZnxBigAddSmallInplace,
        VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace, VecZnxBigFillDistF64,
        VecZnxBigFillNormal, VecZnxBigFromBytes, VecZnxBigNegateInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSub, VecZnxBigSubABInplace, VecZnxBigSubBAInplace, VecZnxBigSubSmallA, VecZnxBigSubSmallAInplace,
        VecZnxBigSubSmallB, VecZnxBigSubSmallBInplace,
    },
    layouts::{Backend, Module, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef},
    oep::{
        VecZnxBigAddDistF64Impl, VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl, VecZnxBigAddSmallImpl,
        VecZnxBigAddSmallInplaceImpl, VecZnxBigAllocBytesImpl, VecZnxBigAllocImpl, VecZnxBigAutomorphismImpl,
        VecZnxBigAutomorphismInplaceImpl, VecZnxBigFillDistF64Impl, VecZnxBigFillNormalImpl, VecZnxBigFromBytesImpl,
        VecZnxBigNegateInplaceImpl, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigSubABInplaceImpl,
        VecZnxBigSubBAInplaceImpl, VecZnxBigSubImpl, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallAInplaceImpl,
        VecZnxBigSubSmallBImpl, VecZnxBigSubSmallBInplaceImpl,
    },
};

impl<B> VecZnxBigAlloc<B> for Module<B>
where
    B: Backend + VecZnxBigAllocImpl<B>,
{
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B> {
        B::vec_znx_big_alloc_impl(self.n(), cols, size)
    }
}

impl<B> VecZnxBigFromBytes<B> for Module<B>
where
    B: Backend + VecZnxBigFromBytesImpl<B>,
{
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B> {
        B::vec_znx_big_from_bytes_impl(self.n(), cols, size, bytes)
    }
}

impl<B> VecZnxBigAllocBytes for Module<B>
where
    B: Backend + VecZnxBigAllocBytesImpl<B>,
{
    fn vec_znx_big_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        B::vec_znx_big_alloc_bytes_impl(self.n(), cols, size)
    }
}

impl<B> VecZnxBigAddDistF64<B> for Module<B>
where
    B: Backend + VecZnxBigAddDistF64Impl<B>,
{
    fn vec_znx_big_add_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        B::add_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B> VecZnxBigAddNormal<B> for Module<B>
where
    B: Backend + VecZnxBigAddNormalImpl<B>,
{
    fn vec_znx_big_add_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        B::add_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}

impl<B> VecZnxBigFillDistF64<B> for Module<B>
where
    B: Backend + VecZnxBigFillDistF64Impl<B>,
{
    fn vec_znx_big_fill_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        B::fill_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B> VecZnxBigFillNormal<B> for Module<B>
where
    B: Backend + VecZnxBigFillNormalImpl<B>,
{
    fn vec_znx_big_fill_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        B::fill_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}

impl<B> VecZnxBigAdd<B> for Module<B>
where
    B: Backend + VecZnxBigAddImpl<B>,
{
    fn vec_znx_big_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_add_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigAddInplace<B> for Module<B>
where
    B: Backend + VecZnxBigAddInplaceImpl<B>,
{
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_add_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigAddSmall<B> for Module<B>
where
    B: Backend + VecZnxBigAddSmallImpl<B>,
{
    fn vec_znx_big_add_small<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        B::vec_znx_big_add_small_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigAddSmallInplace<B> for Module<B>
where
    B: Backend + VecZnxBigAddSmallInplaceImpl<B>,
{
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_add_small_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSub<B> for Module<B>
where
    B: Backend + VecZnxBigSubImpl<B>,
{
    fn vec_znx_big_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubABInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubABInplaceImpl<B>,
{
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_ab_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubBAInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubBAInplaceImpl<B>,
{
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_ba_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubSmallA<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallAImpl<B>,
{
    fn vec_znx_big_sub_small_a<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_small_a_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubSmallAInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallAInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_sub_small_a_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubSmallB<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallBImpl<B>,
{
    fn vec_znx_big_sub_small_b<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        B::vec_znx_big_sub_small_b_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubSmallBInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallBInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_sub_small_b_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigNegateInplace<B> for Module<B>
where
    B: Backend + VecZnxBigNegateInplaceImpl<B>,
{
    fn vec_znx_big_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>,
    {
        B::vec_znx_big_negate_inplace_impl(self, a, a_col);
    }
}

impl<B> VecZnxBigNormalizeTmpBytes for Module<B>
where
    B: Backend + VecZnxBigNormalizeTmpBytesImpl<B>,
{
    fn vec_znx_big_normalize_tmp_bytes(&self, n: usize) -> usize {
        B::vec_znx_big_normalize_tmp_bytes_impl(self, n)
    }
}

impl<B> VecZnxBigNormalize<B> for Module<B>
where
    B: Backend + VecZnxBigNormalizeImpl<B>,
{
    fn vec_znx_big_normalize<R, A>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_normalize_impl(self, basek, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxBigAutomorphism<B> for Module<B>
where
    B: Backend + VecZnxBigAutomorphismImpl<B>,
{
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_automorphism_impl(self, k, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigAutomorphismInplace<B> for Module<B>
where
    B: Backend + VecZnxBigAutomorphismInplaceImpl<B>,
{
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>,
    {
        B::vec_znx_big_automorphism_inplace_impl(self, k, a, a_col);
    }
}
