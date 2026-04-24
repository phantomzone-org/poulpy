use crate::{
    api::{
        VecZnxAddAssign, VecZnxAddInto, VecZnxAddNormal, VecZnxAddScalarAssign, VecZnxAddScalarInto, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxCopy, VecZnxFillNormal, VecZnxFillUniform, VecZnxLsh,
        VecZnxLshAddInto, VecZnxLshInplace, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMergeRings, VecZnxMergeRingsTmpBytes,
        VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace, VecZnxMulXpMinusOneInplaceTmpBytes, VecZnxNegate, VecZnxNegateInplace,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshInplace, VecZnxRshSub,
        VecZnxRshTmpBytes, VecZnxSplitRing, VecZnxSplitRingTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSubNegateInplace,
        VecZnxSubScalar, VecZnxSubScalarInplace, VecZnxSwitchRing, VecZnxZero, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxToRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxToMut, VecZnxToRef,
    },
    oep::HalVecZnxImpl,
    source::Source,
};

macro_rules! impl_vec_znx_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalVecZnxImpl<B>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_delegate!(
    VecZnxZero,
    fn vec_znx_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_zero(self, res, res_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxZeroBackend<B>,
    fn vec_znx_zero_backend<'r>(&self, res: &mut VecZnxBackendMut<'r, B>, res_col: usize) {
        B::vec_znx_zero_backend(self, res, res_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeTmpBytes,
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_normalize_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalize<B>,
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeInplace<B>,
    fn vec_znx_normalize_inplace<'s, A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_normalize_assign(self, base2k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeInplaceBackend<B>,
    fn vec_znx_normalize_inplace_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize_inplace_backend(self, base2k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddInto,
    fn vec_znx_add_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_add_into(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddAssign,
    fn vec_znx_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_add_assign(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarInto,
    fn vec_znx_add_scalar_into<R, A, D>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_add_scalar_into(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssign,
    fn vec_znx_add_scalar_assign<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_add_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSub,
    fn vec_znx_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_sub(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubInplace,
    fn vec_znx_sub_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_assign(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubNegateInplace,
    fn vec_znx_sub_negate_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_negate_assign(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalar,
    fn vec_znx_sub_scalar<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_sub_scalar(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarInplace,
    fn vec_znx_sub_scalar_inplace<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_sub_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegate,
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_negate(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateInplace,
    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_negate_assign(self, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshTmpBytes,
    fn vec_znx_rsh_tmp_bytes(&self) -> usize {
        B::vec_znx_rsh_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshTmpBytes,
    fn vec_znx_lsh_tmp_bytes(&self) -> usize {
        B::vec_znx_lsh_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxLsh<B>,
    fn vec_znx_lsh<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAddInto<B>,
    fn vec_znx_lsh_add_into<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh_add_into(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRsh<B>,
    fn vec_znx_rsh<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAddInto<B>,
    fn vec_znx_rsh_add_into<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh_add_into(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshSub<B>,
    fn vec_znx_lsh_sub<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh_sub(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshSub<B>,
    fn vec_znx_rsh_sub<'s, R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh_sub(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshInplace<B>,
    fn vec_znx_lsh_inplace<'s, A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_lsh_assign(self, base2k, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshInplace<B>,
    fn vec_znx_rsh_inplace<'s, A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rsh_assign(self, base2k, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotate<B>,
    fn vec_znx_rotate<'r, 'a>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_rotate(self, k, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateInplaceTmpBytes,
    fn vec_znx_rotate_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_inplace_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateInplace<B>,
    fn vec_znx_rotate_inplace<'s, 'r>(
        &self,
        k: i64,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rotate_inplace(self, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphism,
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_automorphism(self, k, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismInplaceTmpBytes,
    fn vec_znx_automorphism_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_automorphism_inplace_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismInplace<B>,
    fn vec_znx_automorphism_inplace<'s, 'r>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_automorphism_inplace(self, k, res, res_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOne,
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_mul_xp_minus_one(self, p, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneInplaceTmpBytes,
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes(&self) -> usize {
        B::vec_znx_mul_xp_minus_one_inplace_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneInplace<B>,
    fn vec_znx_mul_xp_minus_one_inplace<'s, R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_mul_xp_minus_one_inplace(self, p, res, res_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxSplitRingTmpBytes,
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize {
        B::vec_znx_split_ring_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxSplitRing<B>,
    fn vec_znx_split_ring<'s, R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_split_ring(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxMergeRingsTmpBytes,
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize {
        B::vec_znx_merge_rings_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxMergeRings<B>,
    fn vec_znx_merge_rings<'s, R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize, scratch: &mut ScratchArena<'s, B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_merge_rings(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxSwitchRing,
    fn vec_znx_switch_ring<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_switch_ring(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxCopy,
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_copy(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniform,
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_uniform(self, base2k, res, res_col, source);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillNormal,
    fn vec_znx_fill_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_normal(self, base2k, res, res_col, noise_infos, source_xe);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormal,
    fn vec_znx_add_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_add_normal(self, base2k, res, res_col, noise_infos, source_xe);
    }
);
