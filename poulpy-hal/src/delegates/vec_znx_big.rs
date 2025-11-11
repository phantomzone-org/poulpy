use crate::{
    api::{
        VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddNormal, VecZnxBigAddSmall, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace, VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigBytesOf,
        VecZnxBigFromBytes, VecZnxBigFromSmall, VecZnxBigNegate, VecZnxBigNegateInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubInplace, VecZnxBigSubNegateInplace, VecZnxBigSubSmallA,
        VecZnxBigSubSmallB, VecZnxBigSubSmallInplace, VecZnxBigSubSmallNegateInplace,
    },
    layouts::{Backend, Module, Scratch, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef},
    oep::{
        VecZnxBigAddImpl, VecZnxBigAddInplaceImpl, VecZnxBigAddNormalImpl, VecZnxBigAddSmallImpl, VecZnxBigAddSmallInplaceImpl,
        VecZnxBigAllocBytesImpl, VecZnxBigAllocImpl, VecZnxBigAutomorphismImpl, VecZnxBigAutomorphismInplaceImpl,
        VecZnxBigAutomorphismInplaceTmpBytesImpl, VecZnxBigFromBytesImpl, VecZnxBigFromSmallImpl, VecZnxBigNegateImpl,
        VecZnxBigNegateInplaceImpl, VecZnxBigNormalizeImpl, VecZnxBigNormalizeTmpBytesImpl, VecZnxBigSubImpl,
        VecZnxBigSubInplaceImpl, VecZnxBigSubNegateInplaceImpl, VecZnxBigSubSmallAImpl, VecZnxBigSubSmallBImpl,
        VecZnxBigSubSmallInplaceImpl, VecZnxBigSubSmallNegateInplaceImpl,
    },
    source::Source,
};

impl<B> VecZnxBigFromSmall<B> for Module<B>
where
    B: Backend + VecZnxBigFromSmallImpl<B>,
{
    fn vec_znx_big_from_small<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_from_small_impl(res, res_col, a, a_col);
    }
}

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

impl<B> VecZnxBigBytesOf for Module<B>
where
    B: Backend + VecZnxBigAllocBytesImpl,
{
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        B::vec_znx_big_bytes_of_impl(self.n(), cols, size)
    }
}

impl<B> VecZnxBigAddNormal<B> for Module<B>
where
    B: Backend + VecZnxBigAddNormalImpl<B>,
{
    fn vec_znx_big_add_normal<R: VecZnxBigToMut<B>>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) {
        B::add_normal_impl(self, base2k, res, res_col, k, source, sigma, bound);
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

impl<B> VecZnxBigSubInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubInplaceImpl<B>,
{
    fn vec_znx_big_sub_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubNegateInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubNegateInplaceImpl<B>,
{
    fn vec_znx_big_sub_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_sub_negate_inplace_impl(self, res, res_col, a, a_col);
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

impl<B> VecZnxBigSubSmallInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_sub_small_inplace_impl(self, res, res_col, a, a_col);
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

impl<B> VecZnxBigSubSmallNegateInplace<B> for Module<B>
where
    B: Backend + VecZnxBigSubSmallNegateInplaceImpl<B>,
{
    fn vec_znx_big_sub_small_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        B::vec_znx_big_sub_small_negate_inplace_impl(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigNegate<B> for Module<B>
where
    B: Backend + VecZnxBigNegateImpl<B>,
{
    fn vec_znx_big_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_negate_impl(self, res, res_col, a, a_col);
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
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_big_normalize_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxBigNormalize<B> for Module<B>
where
    B: Backend + VecZnxBigNormalizeImpl<B>,
{
    fn vec_znx_big_normalize<R, A>(
        &self,
        res_basek: usize,
        res: &mut R,
        res_col: usize,
        a_basek: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        B::vec_znx_big_normalize_impl(self, res_basek, res, res_col, a_basek, a, a_col, scratch);
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

impl<B> VecZnxBigAutomorphismInplaceTmpBytes for Module<B>
where
    B: Backend + VecZnxBigAutomorphismInplaceTmpBytesImpl<B>,
{
    fn vec_znx_big_automorphism_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_big_automorphism_inplace_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxBigAutomorphismInplace<B> for Module<B>
where
    B: Backend + VecZnxBigAutomorphismInplaceImpl<B>,
{
    fn vec_znx_big_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxBigToMut<B>,
    {
        B::vec_znx_big_automorphism_inplace_impl(self, k, a, a_col, scratch);
    }
}
