use sampling::source::Source;

use crate::hal::{
    api::{
        VecZnxAdd, VecZnxAddDistF64, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxCopy, VecZnxFillDistF64, VecZnxFillNormal, VecZnxFillUniform, VecZnxLshInplace,
        VecZnxMerge, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxNegate, VecZnxNegateInplace, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace, VecZnxSplit,
        VecZnxSub, VecZnxSubABInplace, VecZnxSubBAInplace, VecZnxSubScalarInplace, VecZnxSwithcDegree,
    },
    layouts::{Backend, Module, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    oep::{
        VecZnxAddDistF64Impl, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarInplaceImpl,
        VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxCopyImpl, VecZnxFillDistF64Impl, VecZnxFillNormalImpl,
        VecZnxFillUniformImpl, VecZnxLshInplaceImpl, VecZnxMergeImpl, VecZnxMulXpMinusOneImpl, VecZnxMulXpMinusOneInplaceImpl,
        VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl, VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl,
        VecZnxRotateImpl, VecZnxRotateInplaceImpl, VecZnxRshInplaceImpl, VecZnxSplitImpl, VecZnxSubABInplaceImpl,
        VecZnxSubBAInplaceImpl, VecZnxSubImpl, VecZnxSubScalarInplaceImpl, VecZnxSwithcDegreeImpl,
    },
};

impl<B> VecZnxNormalizeTmpBytes for Module<B>
where
    B: Backend + VecZnxNormalizeTmpBytesImpl<B>,
{
    fn vec_znx_normalize_tmp_bytes(&self, n: usize) -> usize {
        B::vec_znx_normalize_tmp_bytes_impl(self, n)
    }
}

impl<B> VecZnxNormalize<B> for Module<B>
where
    B: Backend + VecZnxNormalizeImpl<B>,
{
    fn vec_znx_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_normalize_impl(self, basek, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxNormalizeInplace<B> for Module<B>
where
    B: Backend + VecZnxNormalizeInplaceImpl<B>,
{
    fn vec_znx_normalize_inplace<A>(&self, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_normalize_inplace_impl(self, basek, a, a_col, scratch)
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

impl<B> VecZnxSubABInplace for Module<B>
where
    B: Backend + VecZnxSubABInplaceImpl<B>,
{
    fn vec_znx_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_ab_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSubBAInplace for Module<B>
where
    B: Backend + VecZnxSubBAInplaceImpl<B>,
{
    fn vec_znx_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_ba_inplace_impl(self, res, res_col, a, a_col)
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

impl<B> VecZnxLshInplace for Module<B>
where
    B: Backend + VecZnxLshInplaceImpl<B>,
{
    fn vec_znx_lsh_inplace<A>(&self, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_lsh_inplace_impl(self, basek, k, a)
    }
}

impl<B> VecZnxRshInplace for Module<B>
where
    B: Backend + VecZnxRshInplaceImpl<B>,
{
    fn vec_znx_rsh_inplace<A>(&self, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rsh_inplace_impl(self, basek, k, a)
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

impl<B> VecZnxRotateInplace for Module<B>
where
    B: Backend + VecZnxRotateInplaceImpl<B>,
{
    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rotate_inplace_impl(self, k, a, a_col)
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

impl<B> VecZnxAutomorphismInplace for Module<B>
where
    B: Backend + VecZnxAutomorphismInplaceImpl<B>,
{
    fn vec_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_automorphism_inplace_impl(self, k, a, a_col)
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

impl<B> VecZnxMulXpMinusOneInplace for Module<B>
where
    B: Backend + VecZnxMulXpMinusOneInplaceImpl<B>,
{
    fn vec_znx_mul_xp_minus_one_inplace<R>(&self, p: i64, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_mul_xp_minus_one_inplace_impl(self, p, res, res_col);
    }
}

impl<B> VecZnxSplit<B> for Module<B>
where
    B: Backend + VecZnxSplitImpl<B>,
{
    fn vec_znx_split<R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_split_impl(self, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxMerge for Module<B>
where
    B: Backend + VecZnxMergeImpl<B>,
{
    fn vec_znx_merge<R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_merge_impl(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSwithcDegree for Module<B>
where
    B: Backend + VecZnxSwithcDegreeImpl<B>,
{
    fn vec_znx_switch_degree<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_switch_degree_impl(self, res, res_col, a, a_col)
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
    fn vec_znx_fill_uniform<R>(&self, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_uniform_impl(self, basek, res, res_col, k, source);
    }
}

impl<B> VecZnxFillDistF64 for Module<B>
where
    B: Backend + VecZnxFillDistF64Impl<B>,
{
    fn vec_znx_fill_dist_f64<R, D: rand::prelude::Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B> VecZnxAddDistF64 for Module<B>
where
    B: Backend + VecZnxAddDistF64Impl<B>,
{
    fn vec_znx_add_dist_f64<R, D: rand::prelude::Distribution<f64>>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_add_dist_f64_impl(self, basek, res, res_col, k, source, dist, bound);
    }
}

impl<B> VecZnxFillNormal for Module<B>
where
    B: Backend + VecZnxFillNormalImpl<B>,
{
    fn vec_znx_fill_normal<R>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}

impl<B> VecZnxAddNormal for Module<B>
where
    B: Backend + VecZnxAddNormalImpl<B>,
{
    fn vec_znx_add_normal<R>(
        &self,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        B::vec_znx_add_normal_impl(self, basek, res, res_col, k, source, sigma, bound);
    }
}
