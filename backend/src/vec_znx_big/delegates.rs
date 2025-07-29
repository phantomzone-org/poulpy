use rand_distr::Distribution;
use sampling::source::Source;

use crate::{
    Backend, Module, Scratch, VecZnxBigAdd, VecZnxBigAddDistF64, VecZnxBigAddDistF64Impl, VecZnxBigAddImpl, VecZnxBigAddInplace,
    VecZnxBigAddInplaceImpl, VecZnxBigAddNormal, VecZnxBigAddNormalImpl, VecZnxBigAddSmall, VecZnxBigAddSmallImpl,
    VecZnxBigAddSmallInplace, VecZnxBigAddSmallInplaceImpl, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAllocBytesImpl,
    VecZnxBigAllocImpl, VecZnxBigAutomorphism, VecZnxBigAutomorphismImpl, VecZnxBigAutomorphismInplace,
    VecZnxBigAutomorphismInplaceImpl, VecZnxBigFillDistF64, VecZnxBigFillDistF64Impl, VecZnxBigFillNormal,
    VecZnxBigFillNormalImpl, VecZnxBigFromBytes, VecZnxBigFromBytesImpl, VecZnxBigNegateInplace, VecZnxBigNegateInplaceImpl,
    VecZnxBigNormalize, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytes, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigOwned,
    VecZnxBigSub, VecZnxBigSubABInplace, VecZnxBigSubABInplaceImpl, VecZnxBigSubBAInplace, VecZnxBigSubBAInplaceImpl,
    VecZnxBigSubImpl, VecZnxBigSubSmallA, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallAInplace, VecZnxBigSubSmallAInplaceImpl,
    VecZnxBigSubSmallB, VecZnxBigSubSmallBImpl, VecZnxBigSubSmallBInplace, VecZnxBigSubSmallBInplaceImpl, VecZnxBigToMut,
    VecZnxBigToRef, VecZnxToMut, VecZnxToRef,
};

impl<B: Backend> VecZnxBigAlloc<B> for Module<B>
where
    (): VecZnxBigAllocImpl<B>,
{
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B> {
        <() as VecZnxBigAllocImpl<B>>::vec_znx_big_alloc_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxBigFromBytes<B> for Module<B>
where
    (): VecZnxBigFromBytesImpl<B>,
{
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B> {
        <() as VecZnxBigFromBytesImpl<B>>::vec_znx_big_from_bytes_impl(self, cols, size, bytes)
    }
}

impl<B: Backend> VecZnxBigAllocBytes for Module<B>
where
    (): VecZnxBigAllocBytesImpl<B>,
{
    fn vec_znx_big_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        <() as VecZnxBigAllocBytesImpl<B>>::vec_znx_big_alloc_bytes_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxBigAddDistF64<B> for Module<B>
where
    (): VecZnxBigAddDistF64Impl<B>,
{
    fn add_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        <() as VecZnxBigAddDistF64Impl<B>>::add_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B: Backend> VecZnxBigAddNormal<B> for Module<B>
where
    (): VecZnxBigAddNormalImpl<B>,
{
    fn add_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        <() as VecZnxBigAddNormalImpl<B>>::add_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}

impl<B: Backend> VecZnxBigFillDistF64<B> for Module<B>
where
    (): VecZnxBigFillDistF64Impl<B>,
{
    fn fill_dist_f64<R: VecZnxBigToMut<B>, D: Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) {
        <() as VecZnxBigFillDistF64Impl<B>>::fill_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B: Backend> VecZnxBigFillNormal<B> for Module<B>
where
    (): VecZnxBigFillNormalImpl<B>,
{
    fn fill_normal<R: VecZnxBigToMut<B>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        <() as VecZnxBigFillNormalImpl<B>>::fill_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}

impl<B: Backend> VecZnxBigAdd<B> for Module<B>
where
    (): VecZnxBigAddImpl<B>,
{
    fn vec_znx_big_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigAddImpl<B>>::vec_znx_big_add_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxBigAddInplace<B> for Module<B>
where
    (): VecZnxBigAddInplaceImpl<B>,
{
    fn vec_znx_big_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigAddInplaceImpl<B>>::vec_znx_big_add_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigAddSmall<B> for Module<B>
where
    (): VecZnxBigAddSmallImpl<B>,
{
    fn vec_znx_big_add_small<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        <() as VecZnxBigAddSmallImpl<B>>::vec_znx_big_add_small_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxBigAddSmallInplace<B> for Module<B>
where
    (): VecZnxBigAddSmallInplaceImpl<B>,
{
    fn vec_znx_big_add_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <() as VecZnxBigAddSmallInplaceImpl<B>>::vec_znx_big_add_small_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigSub<B> for Module<B>
where
    (): VecZnxBigSubImpl<B>,
{
    fn vec_znx_big_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigSubImpl<B>>::vec_znx_big_sub_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxBigSubABInplace<B> for Module<B>
where
    (): VecZnxBigSubABInplaceImpl<B>,
{
    fn vec_znx_big_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigSubABInplaceImpl<B>>::vec_znx_big_sub_ab_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigSubBAInplace<B> for Module<B>
where
    (): VecZnxBigSubBAInplaceImpl<B>,
{
    fn vec_znx_big_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigSubBAInplaceImpl<B>>::vec_znx_big_sub_ba_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigSubSmallA<B> for Module<B>
where
    (): VecZnxBigSubSmallAImpl<B>,
{
    fn vec_znx_big_sub_small_a<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigSubSmallAImpl<B>>::vec_znx_big_sub_small_a_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxBigSubSmallAInplace<B> for Module<B>
where
    (): VecZnxBigSubSmallAInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_a_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <() as VecZnxBigSubSmallAInplaceImpl<B>>::vec_znx_big_sub_small_a_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigSubSmallB<B> for Module<B>
where
    (): VecZnxBigSubSmallBImpl<B>,
{
    fn vec_znx_big_sub_small_b<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        <() as VecZnxBigSubSmallBImpl<B>>::vec_znx_big_sub_small_b_impl(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B: Backend> VecZnxBigSubSmallBInplace<B> for Module<B>
where
    (): VecZnxBigSubSmallBInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_b_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <() as VecZnxBigSubSmallBInplaceImpl<B>>::vec_znx_big_sub_small_b_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigNegateInplace<B> for Module<B>
where
    (): VecZnxBigNegateInplaceImpl<B>,
{
    fn vec_znx_big_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>,
    {
        <() as VecZnxBigNegateInplaceImpl<B>>::vec_znx_big_negate_inplace_impl(self, a, a_col);
    }
}

impl<B: Backend> VecZnxBigNormalizeTmpBytes for Module<B>
where
    (): VecZnxBigNormalizeTmpBytesImpl<B>,
{
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        <() as VecZnxBigNormalizeTmpBytesImpl<B>>::vec_znx_big_normalize_tmp_bytes_impl(self)
    }
}

impl<B: Backend> VecZnxBigNormalize<B> for Module<B>
where
    (): VecZnxBigNormalizeImpl<B>,
{
    fn vec_znx_big_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigNormalizeImpl<B>>::vec_znx_big_normalize_impl(self, basek, res, res_col, a, a_col, scratch);
    }
}

impl<B: Backend> VecZnxBigAutomorphism<B> for Module<B>
where
    (): VecZnxBigAutomorphismImpl<B>,
{
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <() as VecZnxBigAutomorphismImpl<B>>::vec_znx_big_automorphism_impl(self, k, res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigAutomorphismInplace<B> for Module<B>
where
    (): VecZnxBigAutomorphismInplaceImpl<B>,
{
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>,
    {
        <() as VecZnxBigAutomorphismInplaceImpl<B>>::vec_znx_big_automorphism_inplace_impl(self, k, a, a_col);
    }
}
