use crate::{
    api::{
        VecZnxAddAssignBackend, VecZnxAddIntoBackend, VecZnxAddNormal, VecZnxAddNormalBackend, VecZnxAddScalarAssign,
        VecZnxAddScalarAssignBackend, VecZnxAddScalarInto, VecZnxAddScalarIntoBackend, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxAutomorphismInplaceTmpBytes, VecZnxCopyBackend, VecZnxFillNormal,
        VecZnxFillNormalBackend, VecZnxFillUniform, VecZnxFillUniformBackend, VecZnxLsh, VecZnxLshAddInto, VecZnxLshInplace,
        VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMergeRings, VecZnxMergeRingsTmpBytes, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneInplace, VecZnxMulXpMinusOneInplaceTmpBytes, VecZnxNegateBackend, VecZnxNegateInplaceBackend,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeInplaceBackend, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshInplace, VecZnxRshSub,
        VecZnxRshTmpBytes, VecZnxSplitRing, VecZnxSplitRingTmpBytes, VecZnxSubBackend, VecZnxSubInplaceBackend,
        VecZnxSubNegateInplaceBackend, VecZnxSubScalar, VecZnxSubScalarBackend, VecZnxSubScalarInplace,
        VecZnxSubScalarInplaceBackend, VecZnxSwitchRingBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnx, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToMut, VecZnxToRef,
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
    VecZnxAddIntoBackend<B>,
    fn vec_znx_add_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    ) {
        B::vec_znx_add_into_backend(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddAssignBackend<B>,
    fn vec_znx_add_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_add_assign_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarInto,
    fn vec_znx_add_scalar_into<R, D>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &ScalarZnx<&[u8]>,
        a_col: usize,
        b: &D,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        D: VecZnxToRef,
    {
        B::vec_znx_add_scalar_into(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarIntoBackend<B>,
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    ) {
        B::vec_znx_add_scalar_into_backend(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssign,
    fn vec_znx_add_scalar_assign<R>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_add_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssignBackend<B>,
    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_add_scalar_assign_backend(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubBackend<B>,
    fn vec_znx_sub_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    ) {
        B::vec_znx_sub_backend(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubInplaceBackend<B>,
    fn vec_znx_sub_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_inplace_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubNegateInplaceBackend<B>,
    fn vec_znx_sub_negate_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_negate_inplace_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalar,
    fn vec_znx_sub_scalar<R, D>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &ScalarZnx<&[u8]>,
        a_col: usize,
        b: &D,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        D: VecZnxToRef,
    {
        B::vec_znx_sub_scalar(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarBackend<B>,
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    ) {
        B::vec_znx_sub_scalar_backend(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarInplace,
    fn vec_znx_sub_scalar_inplace<R>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_sub_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarInplaceBackend<B>,
    fn vec_znx_sub_scalar_inplace_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_scalar_inplace_backend(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateBackend<B>,
    fn vec_znx_negate_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_negate_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateInplaceBackend<B>,
    fn vec_znx_negate_inplace_backend(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize) {
        B::vec_znx_negate_inplace_backend(self, a, a_col)
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
    VecZnxSwitchRingBackend<B>,
    fn vec_znx_switch_ring_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_switch_ring_backend(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxCopyBackend<B>,
    fn vec_znx_copy_backend(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_copy_backend(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniform,
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_uniform_seed(self, base2k, res, res_col, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformBackend<B>,
    fn vec_znx_fill_uniform_backend(&self, base2k: usize, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, seed: [u8; 32]) {
        B::vec_znx_fill_uniform_backend(self, base2k, res, res_col, seed);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillNormal,
    fn vec_znx_fill_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_normal_seed(self, base2k, res, res_col, noise_infos, source_xe.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxFillNormalBackend<B>,
    fn vec_znx_fill_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        B::vec_znx_fill_normal_backend(self, base2k, res, res_col, noise_infos, seed);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormal,
    fn vec_znx_add_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_add_normal_seed(self, base2k, res, res_col, noise_infos, source_xe.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormalBackend<B>,
    fn vec_znx_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        B::vec_znx_add_normal_backend(self, base2k, res, res_col, noise_infos, seed);
    }
);
