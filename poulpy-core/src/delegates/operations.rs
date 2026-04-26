use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, Module, ScratchArena};

use crate::{
    api::{
        GGSWRotate, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENormalize, GLWEPackerOps, GLWEPacking, GLWERotate,
        GLWEShift, GLWETensoring, GLWETrace,
    },
    glwe_packer::GLWEPackerOpsDefault,
    glwe_packing::GLWEPackingDefault,
    glwe_trace::GLWETraceDefault,
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWETensor,
        GLWETensorKeyPrepared, GLWEToBackendMut, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    oep::{
        GGSWRotateImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENormalizeImpl, GLWEPackImpl,
        GLWERotateImpl, GLWEShiftImpl, GLWETensoringImpl, GLWETraceImpl,
    },
    operations::{
        GGSWRotateDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault, GLWENormalizeDefault,
        GLWERotateDefault, GLWEShiftDefault, GLWETensoringDefault,
    },
};

macro_rules! impl_operations_delegate {
    ($trait:ty, $impl_trait:path, $default:path, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + $impl_trait,
            Module<BE>: $default,
        {
            $($body)+
        }
    };
}

impl_operations_delegate!(
    GLWEMulConst<BE>,
    GLWEMulConstImpl<BE>,
    GLWEMulConstDefault<BE>,
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_mul_const_tmp_bytes(self, res, a, b_size)
    },
    fn glwe_mul_const<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_mul_const(self, cnv_offset, res, a, b, scratch)
    },
    fn glwe_mul_const_inplace<'s, R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut ScratchArena<'s, BE>)
    where
        R: HostDataMut,
        GLWE<R>: GLWEToBackendMut<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_mul_const_assign(self, cnv_offset, res, b, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulPlain<BE>,
    GLWEMulPlainImpl<BE>,
    GLWEMulPlainDefault<BE>,
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_plain_tmp_bytes(self, res, a, b)
    },
    fn glwe_mul_plain<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        B: HostDataRef,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: crate::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_mul_plain(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    },
    fn glwe_mul_plain_inplace<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWEPlaintext<A>: crate::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_mul_plain_assign(self, cnv_offset, res, res_effective_k, a, a_effective_k, scratch)
    }
);

impl_operations_delegate!(
    GLWETensoring<BE>,
    GLWETensoringImpl<BE>,
    GLWETensoringDefault<BE>,
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_tensor_apply_tmp_bytes(self, res, a, b)
    },
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_tensor_square_apply_tmp_bytes(self, res, a)
    },
    fn glwe_tensor_apply<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        B: HostDataRef,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWE<B>: crate::layouts::GLWEToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_tensor_apply(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    },
    fn glwe_tensor_square_apply<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_tensor_square_apply(self, cnv_offset, res, a, a_effective_k, scratch)
    },
    fn glwe_tensor_relinearize<'s, R, A, B>(
        &self,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: HostDataMut,
        A: HostDataRef,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: crate::layouts::GLWEToBackendRef<BE>,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        BE::glwe_tensor_relinearize(self, res, a, tsk, tsk_size, scratch)
    },
    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_tensor_relinearize_tmp_bytes(self, res, a, tsk)
    }
);

impl_operations_delegate!(
    GLWERotate<BE>,
    GLWERotateImpl<BE>,
    GLWERotateDefault<BE>,
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        BE::glwe_rotate_tmp_bytes(self)
    },
    fn glwe_rotate<'r, 'a>(&self, k: i64, res: &mut GLWEBackendMut<'r, BE>, a: &GLWEBackendRef<'a, BE>) {
        BE::glwe_rotate(self, k, res, a)
    },
    fn glwe_rotate_inplace<'s, 'r>(&self, k: i64, res: &mut GLWEBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_rotate_inplace(self, k, res, scratch);
    }
);

impl_operations_delegate!(
    GGSWRotate<BE>,
    GGSWRotateImpl<BE>,
    GGSWRotateDefault<BE>,
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        BE::ggsw_rotate_tmp_bytes(self)
    },
    fn ggsw_rotate<'r, 'a>(
        &self,
        k: i64,
        res: &mut crate::layouts::GGSWBackendMut<'r, BE>,
        a: &crate::layouts::GGSWBackendRef<'a, BE>,
    ) {
        BE::ggsw_rotate(self, k, res, a)
    },
    fn ggsw_rotate_inplace<'s, 'r>(
        &self,
        k: i64,
        res: &mut crate::layouts::GGSWBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE> + poulpy_hal::api::ScratchAvailable,
    {
        BE::ggsw_rotate_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulXpMinusOne<BE>,
    GLWEMulXpMinusOneImpl<BE>,
    GLWEMulXpMinusOneDefault<BE>,
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        A: crate::layouts::GLWEToBackendRef<BE>,
    {
        BE::glwe_mul_xp_minus_one(self, k, res, a)
    },
    fn glwe_mul_xp_minus_one_inplace<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
    {
        BE::glwe_mul_xp_minus_one_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEShift<BE>,
    GLWEShiftImpl<BE>,
    GLWEShiftDefault<BE>,
    fn glwe_shift_tmp_bytes(&self) -> usize {
        BE::glwe_shift_tmp_bytes(self)
    },
    fn glwe_rsh<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_rsh(self, k, res, scratch)
    },
    fn glwe_lsh_inplace<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_inplace(self, res, k, scratch)
    },
    fn glwe_lsh<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        A: crate::layouts::GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh(self, res, a, k, scratch)
    },
    fn glwe_lsh_add<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        A: crate::layouts::GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_add(self, res, a, k, scratch)
    },
    fn glwe_lsh_sub<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        A: crate::layouts::GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_sub(self, res, a, k, scratch)
    }
);

impl_operations_delegate!(
    GLWENormalize<BE>,
    GLWENormalizeImpl<BE>,
    GLWENormalizeDefault<BE>,
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_normalize_tmp_bytes(self)
    },
    fn glwe_maybe_cross_normalize_to_ref<'a>(
        &self,
        glwe: &'a GLWEBackendRef<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendRef<'a, BE>
    where
        ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        BE::glwe_maybe_cross_normalize_to_ref(self, glwe, target_base2k, tmp_slot, scratch)
    },
    fn glwe_maybe_cross_normalize_to_mut<'a>(
        &self,
        glwe: &'a mut GLWEBackendMut<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendMut<'a, BE>
    where
        ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        BE::glwe_maybe_cross_normalize_to_mut(self, glwe, target_base2k, tmp_slot, scratch)
    },
    fn glwe_normalize<'s, 'r, 'a>(
        &self,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_normalize(self, res, a, scratch)
    },
    fn glwe_normalize_inplace<'s, 'r>(&self, res: &mut GLWEBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>)
    where
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_normalize_assign(self, res, scratch)
    }
);

impl_operations_delegate!(
    GLWETrace<BE>,
    GLWETraceImpl<BE>,
    GLWETraceDefault<BE>,
    fn glwe_trace_galois_elements(&self) -> Vec<i64> {
        BE::glwe_trace_galois_elements(self)
    },
    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_trace_tmp_bytes(self, res_infos, a_infos, key_infos)
    },
    fn glwe_trace<'s, R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_trace(self, res, skip, a, keys, scratch)
    },
    fn glwe_trace_inplace<'s, R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_trace_assign(self, res, skip, keys, scratch)
    }
);

impl_operations_delegate!(
    GLWEPacking<BE>,
    GLWEPackImpl<BE>,
    GLWEPackingDefault<BE>,
    fn glwe_pack_galois_elements(&self) -> Vec<i64> {
        BE::glwe_pack_galois_elements(self)
    },
    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_pack_tmp_bytes(self, res, key)
    },
    fn glwe_pack<'s, R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        GLWE<Vec<u8>>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>,
        BE: 's,
    {
        BE::glwe_pack(self, res, a, log_gap_out, keys, scratch)
    }
);

impl_operations_delegate!(
    GLWEPackerOps<BE>,
    GLWEPackImpl<BE>,
    GLWEPackerOpsDefault<BE>,
    fn packer_add<'s, A, K, H>(
        &self,
        packer: &mut crate::GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        GLWE<Vec<u8>>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>,
        BE: 's,
    {
        BE::packer_add(self, packer, a, i, auto_keys, scratch)
    }
);
