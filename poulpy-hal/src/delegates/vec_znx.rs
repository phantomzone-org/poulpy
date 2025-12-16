use crate::{
    api::{
        VecZnxAdd, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalar, VecZnxAddScalarInplace, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxCopy, VecZnxFillNormal, VecZnxFillUniform, VecZnxLsh,
        VecZnxLshInplace, VecZnxLshTmpBytes, VecZnxMergeRings, VecZnxMergeRingsTmpBytes, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneInplace, VecZnxMulXpMinusOneInplaceTmpBytes, VecZnxNegate, VecZnxNegateInplace, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes,
        VecZnxRsh, VecZnxRshInplace, VecZnxRshTmpBytes, VecZnxSplitRing, VecZnxSplitRingTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSubNegateInplace, VecZnxSubScalar, VecZnxSubScalarInplace, VecZnxSwitchRing, VecZnxZero,
    },
    layouts::{Backend, Module, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    oep::{
        VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarImpl, VecZnxAddScalarInplaceImpl,
        VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxAutomorphismInplaceTmpBytesImpl, VecZnxCopyImpl,
        VecZnxFillNormalImpl, VecZnxFillUniformImpl, VecZnxLshImpl, VecZnxLshInplaceImpl, VecZnxLshTmpBytesImpl,
        VecZnxMergeRingsImpl, VecZnxMergeRingsTmpBytesImpl, VecZnxMulXpMinusOneImpl, VecZnxMulXpMinusOneInplaceImpl,
        VecZnxMulXpMinusOneInplaceTmpBytesImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
        VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
        VecZnxRotateInplaceTmpBytesImpl, VecZnxRshImpl, VecZnxRshInplaceImpl, VecZnxRshTmpBytesImpl, VecZnxSplitRingImpl,
        VecZnxSplitRingTmpBytesImpl, VecZnxSubImpl, VecZnxSubInplaceImpl, VecZnxSubNegateInplaceImpl, VecZnxSubScalarImpl,
        VecZnxSubScalarInplaceImpl, VecZnxSwitchRingImpl, VecZnxZeroImpl,
    },
    source::Source,
};

impl<B> VecZnxZero for Module<B>
where
    B: Backend + VecZnxZeroImpl<B>,
{
    fn vec_znx_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_zero_impl(self, res, res_col);
    }
}

impl<B> VecZnxNormalizeTmpBytes for Module<B>
where
    B: Backend + VecZnxNormalizeTmpBytesImpl<B>,
{
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_normalize_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxNormalize<B> for Module<B>
where
    B: Backend + VecZnxNormalizeImpl<B>,
{
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_normalize_impl(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch)
    }
}

impl<B> VecZnxNormalizeInplace<B> for Module<B>
where
    B: Backend + VecZnxNormalizeInplaceImpl<B>,
{
    fn vec_znx_normalize_inplace<A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_normalize_inplace_impl(self, base2k, a, a_col, scratch)
    }
}

impl<B> VecZnxAdd for Module<B>
where
    B: Backend + VecZnxAddImpl<B>,
{
    fn vec_znx_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_add_impl(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B> VecZnxAddInplace for Module<B>
where
    B: Backend + VecZnxAddInplaceImpl<B>,
{
    fn vec_znx_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_add_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxAddScalar for Module<B>
where
    B: Backend + VecZnxAddScalarImpl<B>,
{
    fn vec_znx_add_scalar<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_add_scalar_impl(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
}

impl<B> VecZnxAddScalarInplace for Module<B>
where
    B: Backend + VecZnxAddScalarInplaceImpl<B>,
{
    fn vec_znx_add_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_add_scalar_inplace_impl(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B> VecZnxSub for Module<B>
where
    B: Backend + VecZnxSubImpl<B>,
{
    fn vec_znx_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_sub_impl(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B> VecZnxSubInplace for Module<B>
where
    B: Backend + VecZnxSubInplaceImpl<B>,
{
    fn vec_znx_sub_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSubNegateInplace for Module<B>
where
    B: Backend + VecZnxSubNegateInplaceImpl<B>,
{
    fn vec_znx_sub_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_negate_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSubScalar for Module<B>
where
    B: Backend + VecZnxSubScalarImpl<B>,
{
    fn vec_znx_sub_scalar<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_sub_scalar_impl(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
}

impl<B> VecZnxSubScalarInplace for Module<B>
where
    B: Backend + VecZnxSubScalarInplaceImpl<B>,
{
    fn vec_znx_sub_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_sub_scalar_inplace_impl(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B> VecZnxNegate for Module<B>
where
    B: Backend + VecZnxNegateImpl<B>,
{
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_negate_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxNegateInplace for Module<B>
where
    B: Backend + VecZnxNegateInplaceImpl<B>,
{
    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_negate_inplace_impl(self, a, a_col)
    }
}

impl<B> VecZnxRshTmpBytes for Module<B>
where
    B: Backend + VecZnxRshTmpBytesImpl<B>,
{
    fn vec_znx_rsh_tmp_bytes(&self) -> usize {
        B::vec_znx_rsh_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxLshTmpBytes for Module<B>
where
    B: Backend + VecZnxLshTmpBytesImpl<B>,
{
    fn vec_znx_lsh_tmp_bytes(&self) -> usize {
        B::vec_znx_lsh_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxLsh<B> for Module<B>
where
    B: Backend + VecZnxLshImpl<B>,
{
    fn vec_znx_lsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh_impl(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxRsh<B> for Module<B>
where
    B: Backend + VecZnxRshImpl<B>,
{
    fn vec_znx_rsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh_impl(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxLshInplace<B> for Module<B>
where
    B: Backend + VecZnxLshInplaceImpl<B>,
{
    fn vec_znx_lsh_inplace<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_lsh_inplace_impl(self, base2k, k, a, a_col, scratch)
    }
}

impl<B> VecZnxRshInplace<B> for Module<B>
where
    B: Backend + VecZnxRshInplaceImpl<B>,
{
    fn vec_znx_rsh_inplace<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rsh_inplace_impl(self, base2k, k, a, a_col, scratch)
    }
}

impl<B> VecZnxRotate for Module<B>
where
    B: Backend + VecZnxRotateImpl<B>,
{
    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rotate_impl(self, k, res, res_col, a, a_col)
    }
}

impl<B> VecZnxRotateInplaceTmpBytes for Module<B>
where
    B: Backend + VecZnxRotateInplaceTmpBytesImpl<B>,
{
    fn vec_znx_rotate_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_inplace_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxRotateInplace<B> for Module<B>
where
    B: Backend + VecZnxRotateInplaceImpl<B>,
{
    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rotate_inplace_impl(self, k, a, a_col, scratch)
    }
}

impl<B> VecZnxAutomorphism for Module<B>
where
    B: Backend + VecZnxAutomorphismImpl<B>,
{
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_automorphism_impl(self, k, res, res_col, a, a_col)
    }
}

impl<B> VecZnxAutomorphismInplaceTmpBytes for Module<B>
where
    B: Backend + VecZnxAutomorphismInplaceTmpBytesImpl<B>,
{
    fn vec_znx_automorphism_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_automorphism_inplace_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxAutomorphismInplace<B> for Module<B>
where
    B: Backend + VecZnxAutomorphismInplaceImpl<B>,
{
    fn vec_znx_automorphism_inplace<R>(&self, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_automorphism_inplace_impl(self, k, res, res_col, scratch)
    }
}

impl<B> VecZnxMulXpMinusOne for Module<B>
where
    B: Backend + VecZnxMulXpMinusOneImpl<B>,
{
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_mul_xp_minus_one_impl(self, p, res, res_col, a, a_col);
    }
}

impl<B> VecZnxMulXpMinusOneInplaceTmpBytes for Module<B>
where
    B: Backend + VecZnxMulXpMinusOneInplaceTmpBytesImpl<B>,
{
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_mul_xp_minus_one_inplace_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxMulXpMinusOneInplace<B> for Module<B>
where
    B: Backend + VecZnxMulXpMinusOneInplaceImpl<B>,
{
    fn vec_znx_mul_xp_minus_one_inplace<R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_mul_xp_minus_one_inplace_impl(self, p, res, res_col, scratch);
    }
}

impl<B> VecZnxSplitRingTmpBytes for Module<B>
where
    B: Backend + VecZnxSplitRingTmpBytesImpl<B>,
{
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize {
        B::vec_znx_split_ring_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxSplitRing<B> for Module<B>
where
    B: Backend + VecZnxSplitRingImpl<B>,
{
    fn vec_znx_split_ring<R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_split_ring_impl(self, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxMergeRingsTmpBytes for Module<B>
where
    B: Backend + VecZnxMergeRingsTmpBytesImpl<B>,
{
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize {
        B::vec_znx_merge_rings_tmp_bytes_impl(self)
    }
}

impl<B> VecZnxMergeRings<B> for Module<B>
where
    B: Backend + VecZnxMergeRingsImpl<B>,
{
    fn vec_znx_merge_rings<R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_merge_rings_impl(self, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxSwitchRing for Module<B>
where
    B: Backend + VecZnxSwitchRingImpl<B>,
{
    fn vec_znx_switch_ring<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_switch_ring_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxCopy for Module<B>
where
    B: Backend + VecZnxCopyImpl<B>,
{
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_copy_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxFillUniform for Module<B>
where
    B: Backend + VecZnxFillUniformImpl<B>,
{
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_uniform_impl(self, base2k, res, res_col, source);
    }
}

impl<B> VecZnxFillNormal for Module<B>
where
    B: Backend + VecZnxFillNormalImpl<B>,
{
    fn vec_znx_fill_normal<R>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_normal_impl(self, base2k, res, res_col, k, source, sigma, bound);
    }
}

impl<B> VecZnxAddNormal for Module<B>
where
    B: Backend + VecZnxAddNormalImpl<B>,
{
    fn vec_znx_add_normal<R>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_add_normal_impl(self, base2k, res, res_col, k, source, sigma, bound);
    }
}
