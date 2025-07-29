use crate::{
    Backend, Module, ScalarZnxToRef, Scratch, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddScalarInplaceImpl,
    VecZnxAllocBytesImpl, VecZnxAllocImpl, VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxCopyImpl,
    VecZnxFromBytesImpl, VecZnxMergeImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
    VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxOwned, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
    VecZnxShiftInplaceImpl, VecZnxSplitImpl, VecZnxSubABInplaceImpl, VecZnxSubBAInplaceImpl, VecZnxSubImpl,
    VecZnxSubScalarInplaceImpl, VecZnxSwithcDegreeImpl, VecZnxToMut, VecZnxToRef,
    vec_znx::traits::{
        VecZnxAdd, VecZnxAddInplace, VecZnxAddScalarInplace, VecZnxAlloc, VecZnxAllocBytes, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxCopy, VecZnxFromBytes, VecZnxMerge, VecZnxNegate, VecZnxNegateInplace, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxShiftInplace, VecZnxSplit,
        VecZnxSub, VecZnxSubABInplace, VecZnxSubBAInplace, VecZnxSubScalarInplace, VecZnxSwithcDegree,
    },
};

impl<B: Backend> VecZnxAlloc for Module<B>
where
    (): VecZnxAllocImpl<B>,
{
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnxOwned {
        <() as VecZnxAllocImpl<B>>::vec_znx_alloc_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxFromBytes for Module<B>
where
    (): VecZnxFromBytesImpl<B>,
{
    fn vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned {
        <() as VecZnxFromBytesImpl<B>>::vec_znx_from_bytes_impl(self, cols, size, bytes)
    }
}

impl<B: Backend> VecZnxAllocBytes for Module<B>
where
    (): VecZnxAllocBytesImpl<B>,
{
    fn vec_znx_alloc_bytes(&self, cols: usize, size: usize) -> usize {
        <() as VecZnxAllocBytesImpl<B>>::vec_znx_alloc_bytes_impl(self, cols, size)
    }
}

impl<B: Backend> VecZnxNormalizeTmpBytes for Module<B>
where
    (): VecZnxNormalizeTmpBytesImpl<B>,
{
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        <() as VecZnxNormalizeTmpBytesImpl<B>>::vec_znx_normalize_tmp_bytes_impl(self)
    }
}

impl<B: Backend> VecZnxNormalize for Module<B>
where
    (): VecZnxNormalizeImpl<B>,
{
    fn vec_znx_normalize<R, A>(&self, basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxNormalizeImpl<B>>::vec_znx_normalize_impl(self, basek, res, res_col, a, a_col, scratch)
    }
}

impl<B: Backend> VecZnxNormalizeInplace for Module<B>
where
    (): VecZnxNormalizeInplaceImpl<B>,
{
    fn vec_znx_normalize_inplace<A>(&self, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut,
    {
        <() as VecZnxNormalizeInplaceImpl<B>>::vec_znx_normalize_inplace_impl(self, basek, a, a_col, scratch)
    }
}

impl<B: Backend> VecZnxAdd for Module<B>
where
    (): VecZnxAddImpl<B>,
{
    fn vec_znx_add<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        <() as VecZnxAddImpl<B>>::vec_znx_add_impl(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B: Backend> VecZnxAddInplace for Module<B>
where
    (): VecZnxAddInplaceImpl<B>,
{
    fn vec_znx_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxAddInplaceImpl<B>>::vec_znx_add_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxAddScalarInplace for Module<B>
where
    (): VecZnxAddScalarInplaceImpl<B>,
{
    fn vec_znx_add_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        <() as VecZnxAddScalarInplaceImpl<B>>::vec_znx_add_scalar_inplace_impl(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B: Backend> VecZnxSub for Module<B>
where
    (): VecZnxSubImpl<B>,
{
    fn vec_znx_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        <() as VecZnxSubImpl<B>>::vec_znx_sub_impl(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B: Backend> VecZnxSubABInplace for Module<B>
where
    (): VecZnxSubABInplaceImpl<B>,
{
    fn vec_znx_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxSubABInplaceImpl<B>>::vec_znx_sub_ab_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxSubBAInplace for Module<B>
where
    (): VecZnxSubBAInplaceImpl<B>,
{
    fn vec_znx_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxSubBAInplaceImpl<B>>::vec_znx_sub_ba_inplace_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxSubScalarInplace for Module<B>
where
    (): VecZnxSubScalarInplaceImpl<B>,
{
    fn vec_znx_sub_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        <() as VecZnxSubScalarInplaceImpl<B>>::vec_znx_sub_scalar_inplace_impl(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B: Backend> VecZnxNegate for Module<B>
where
    (): VecZnxNegateImpl<B>,
{
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxNegateImpl<B>>::vec_znx_negate_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxNegateInplace for Module<B>
where
    (): VecZnxNegateInplaceImpl<B>,
{
    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        <() as VecZnxNegateInplaceImpl<B>>::vec_znx_negate_inplace_impl(self, a, a_col)
    }
}

impl<B: Backend> VecZnxShiftInplace for Module<B>
where
    (): VecZnxShiftInplaceImpl<B>,
{
    fn vec_znx_shift_inplace<A>(&self, basek: usize, k: i64, a: &mut A, scratch: &mut Scratch)
    where
        A: VecZnxToMut,
    {
        <() as VecZnxShiftInplaceImpl<B>>::vec_znx_shift_inplace_impl(self, basek, k, a, scratch)
    }
}

impl<B: Backend> VecZnxRotate for Module<B>
where
    (): VecZnxRotateImpl<B>,
{
    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxRotateImpl<B>>::vec_znx_rotate_impl(self, k, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxRotateInplace for Module<B>
where
    (): VecZnxRotateInplaceImpl<B>,
{
    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        <() as VecZnxRotateInplaceImpl<B>>::vec_znx_rotate_inplace_impl(self, k, a, a_col)
    }
}

impl<B: Backend> VecZnxAutomorphism for Module<B>
where
    (): VecZnxAutomorphismImpl<B>,
{
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxAutomorphismImpl<B>>::vec_znx_automorphism_impl(self, k, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxAutomorphismInplace for Module<B>
where
    (): VecZnxAutomorphismInplaceImpl<B>,
{
    fn vec_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        <() as VecZnxAutomorphismInplaceImpl<B>>::vec_znx_automorphism_inplace_impl(self, k, a, a_col)
    }
}

impl<B: Backend> VecZnxSplit for Module<B>
where
    (): VecZnxSplitImpl<B>,
{
    fn vec_znx_split<R, A>(&self, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxSplitImpl<B>>::vec_znx_split_impl(self, res, res_col, a, a_col, scratch)
    }
}

impl<B: Backend> VecZnxMerge for Module<B>
where
    (): VecZnxMergeImpl<B>,
{
    fn vec_znx_merge<R, A>(&self, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxMergeImpl<B>>::vec_znx_merge_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxSwithcDegree for Module<B>
where
    (): VecZnxSwithcDegreeImpl<B>,
{
    fn vec_znx_switch_degree<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxSwithcDegreeImpl<B>>::vec_znx_switch_degree_impl(self, res, res_col, a, a_col)
    }
}

impl<B: Backend> VecZnxCopy for Module<B>
where
    (): VecZnxCopyImpl<B>,
{
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        <() as VecZnxCopyImpl<B>>::vec_znx_copy_impl(self, res, res_col, a, a_col)
    }
}
