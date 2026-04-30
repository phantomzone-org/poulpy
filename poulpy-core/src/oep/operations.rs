use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    glwe_packer::{GLWEPacker, GLWEPackerOpsDefault},
    glwe_packing::GLWEPackingDefault,
    glwe_trace::GLWETraceDefault,
    layouts::{
        GGLWEInfos, GGSWBackendMut, GGSWBackendRef, GLWE, GLWEAutomorphismKeyHelper, GLWEBackendMut, GLWEBackendRef, GLWEInfos,
        GLWEPlaintext, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    operations::{
        GGSWRotateDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault, GLWENormalizeDefault,
        GLWERotateDefault, GLWEShiftDefault, GLWETensoringDefault,
    },
};

/// Backend-provided GLWE constant-multiplication operations.
///
/// # Safety
/// Implementations must respect the provided layout metadata, conversion offset, and scratch-space
/// contracts, and must not read or write outside the specified backend-owned buffers.
pub unsafe trait GLWEMulConstImpl<BE: Backend>: Backend {
    fn glwe_mul_const_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>;

    fn glwe_mul_const_assign<'s, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>;
}

/// Backend-provided GLWE-by-plaintext multiplication operations.
///
/// # Safety
/// Implementations must interpret the plaintext and ciphertext layouts consistently with the
/// backend and preserve all aliasing and buffer-bound invariants.
pub unsafe trait GLWEMulPlainImpl<BE: Backend>: Backend {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: crate::layouts::GLWEPlaintextToBackendRef<BE>;

    fn glwe_mul_plain_assign<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        GLWEPlaintext<A>: crate::layouts::GLWEPlaintextToBackendRef<BE>;
}

/// Backend-provided GLWE tensoring and relinearization operations.
///
/// # Safety
/// Implementations must preserve tensor layout semantics, respect the temporary-size contracts,
/// and only touch backend-owned storage regions that belong to the supplied operands.
pub unsafe trait GLWETensoringImpl<BE: Backend>: Backend {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWE<B>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_square_apply<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;
}

/// Backend-provided GLWE rotation operations.
///
/// # Safety
/// Implementations must perform rotations according to the polynomial layout without violating
/// scratch-space, aliasing, or buffer-bound guarantees.
pub unsafe trait GLWERotateImpl<BE: Backend>: Backend {
    fn glwe_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rotate<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GLWEBackendMut<'r, BE>, a: &GLWEBackendRef<'a, BE>);

    fn glwe_rotate_assign<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GLWEBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    );
}

/// Backend-provided GGSW rotation operations.
///
/// # Safety
/// Implementations must preserve the GGSW structure for the backend and may only use scratch space
/// and in-place mutation in ways compatible with the advertised contracts.
pub unsafe trait GGSWRotateImpl<BE: Backend>: Backend {
    fn ggsw_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn ggsw_rotate<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GGSWBackendMut<'r, BE>, a: &GGSWBackendRef<'a, BE>);

    fn ggsw_rotate_assign<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GGSWBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: poulpy_hal::api::ScratchAvailable;
}

/// Backend-provided multiplication by `X^p - 1` operations.
///
/// # Safety
/// Implementations must apply the requested ring operation without violating the layout or memory
/// invariants of the supplied ciphertext buffers.
pub unsafe trait GLWEMulXpMinusOneImpl<BE: Backend>: Backend {
    fn glwe_mul_xp_minus_one<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GLWE shift operations.
///
/// # Safety
/// Implementations must respect the polynomial/ciphertext layout and scratch requirements, and may
/// not read or write beyond the backend-owned regions described by the inputs.
pub unsafe trait GLWEShiftImpl<BE: Backend>: Backend {
    fn glwe_shift_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rsh<'s, R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_assign<'s, R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>;

    fn glwe_lsh<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_add<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_sub<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

/// Backend-provided GLWE normalization operations.
///
/// # Safety
/// Implementations must return views that remain valid for the advertised lifetime, preserve
/// normalization semantics, and avoid aliasing or out-of-bounds access across temporary buffers.
pub unsafe trait GLWENormalizeImpl<BE: Backend>: Backend {
    fn glwe_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_maybe_cross_normalize_to_ref<'a>(
        module: &Module<BE>,
        glwe: &'a GLWEBackendRef<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendRef<'a, BE>
    where
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_maybe_cross_normalize_to_mut<'a>(
        module: &Module<BE>,
        glwe: &'a mut GLWEBackendMut<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendMut<'a, BE>
    where
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_normalize<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_normalize_assign<'s, 'r>(module: &Module<BE>, res: &mut GLWEBackendMut<'r, BE>, scratch: &mut ScratchArena<'s, BE>);
}

/// Backend-provided GLWE trace operations.
///
/// # Safety
/// Implementations must apply the requested automorphism sequence faithfully, interpret prepared
/// keys correctly, and keep all accesses within the described ciphertext and scratch regions.
pub unsafe trait GLWETraceImpl<BE: Backend>: Backend {
    fn glwe_trace_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn glwe_trace_assign<'s, R, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

/// Backend-provided GLWE packing operations.
///
/// # Safety
/// Implementations must maintain ciphertext correctness while combining inputs, and must respect
/// all backend buffer, aliasing, and scratch-space invariants expected by the higher layers.
pub unsafe trait GLWEPackImpl<BE: Backend>: Backend {
    fn glwe_pack_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

    fn packer_add<'s, A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker<BE::OwnedBuf>,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        crate::layouts::BackendGLWE<BE>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

#[doc(hidden)]
pub trait OperationsDefaults<BE: Backend>: Backend {
    fn glwe_mul_const_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>;

    fn glwe_mul_const_assign_default<'s, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>;

    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_default<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: crate::layouts::GLWEPlaintextToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        GLWEPlaintext<A>: crate::layouts::GLWEPlaintextToBackendRef<BE>;

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_default<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWE<B>: crate::layouts::GLWEToBackendRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize_default<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: crate::layouts::GLWEToBackendRef<BE>;

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_rotate_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_rotate_default<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GLWEBackendMut<'r, BE>, a: &GLWEBackendRef<'a, BE>);

    fn glwe_rotate_assign_default<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GLWEBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn ggsw_rotate_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn ggsw_rotate_default<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GGSWBackendMut<'r, BE>, a: &GGSWBackendRef<'a, BE>);

    fn ggsw_rotate_assign_default<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GGSWBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: poulpy_hal::api::ScratchAvailable;

    fn glwe_mul_xp_minus_one_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign_default<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;

    fn glwe_shift_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_rsh_default<'s, R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>;

    fn glwe_lsh_assign_default<'s, R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>;

    fn glwe_lsh_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_add_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_sub_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_normalize_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_maybe_cross_normalize_to_ref_default<'a>(
        module: &Module<BE>,
        glwe: &'a GLWEBackendRef<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendRef<'a, BE>;

    fn glwe_maybe_cross_normalize_to_mut_default<'a>(
        module: &Module<BE>,
        glwe: &'a mut GLWEBackendMut<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendMut<'a, BE>;

    fn glwe_normalize_default<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn glwe_normalize_assign_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn glwe_trace_galois_elements_default(module: &Module<BE>) -> Vec<i64>;

    fn glwe_trace_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_default<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>;

    fn glwe_trace_assign_default<'s, R, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>;

    fn glwe_pack_galois_elements_default(module: &Module<BE>) -> Vec<i64>;

    fn glwe_pack_tmp_bytes_default<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack_default<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
        GLWE<Vec<u8>>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>;

    fn packer_add_default<'s, A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker<BE::OwnedBuf>,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: Backend<OwnedBuf = Vec<u8>>;
}

impl<BE: Backend> OperationsDefaults<BE> for BE
where
    Module<BE>: GLWEMulConstDefault<BE>
        + GLWEMulPlainDefault<BE>
        + GLWETensoringDefault<BE>
        + GLWERotateDefault<BE>
        + GGSWRotateDefault<BE>
        + GLWEMulXpMinusOneDefault<BE>
        + GLWEShiftDefault<BE>
        + GLWENormalizeDefault<BE>
        + GLWETraceDefault<BE>
        + GLWEPackingDefault<BE>
        + GLWEPackerOpsDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_mul_const_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const_tmp_bytes(module, res, a, b_size)
    }

    fn glwe_mul_const_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const(module, cnv_offset, res, a, b, scratch)
    }

    fn glwe_mul_const_assign_default<'s, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        b: &[i64],
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const_assign(module, cnv_offset, res, b, scratch)
    }

    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain_tmp_bytes(module, res, a, b)
    }

    fn glwe_mul_plain_default<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWE<A>: GLWEToBackendRef<BE>,
        GLWEPlaintext<B>: crate::layouts::GLWEPlaintextToBackendRef<BE>,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain(
            module,
            cnv_offset,
            res,
            a,
            a_effective_k,
            b,
            b_effective_k,
            scratch,
        )
    }

    fn glwe_mul_plain_assign_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWE<R>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        GLWEPlaintext<A>: crate::layouts::GLWEPlaintextToBackendRef<BE>,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain_assign(
            module,
            cnv_offset,
            res,
            res_effective_k,
            a,
            a_effective_k,
            scratch,
        )
    }

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_apply_tmp_bytes(module, res, a, b)
    }

    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_square_apply_tmp_bytes(module, res, a)
    }

    fn glwe_tensor_apply_default<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
        GLWE<B>: crate::layouts::GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_apply(
            module,
            cnv_offset,
            res,
            a,
            a_effective_k,
            b,
            b_effective_k,
            scratch,
        )
    }

    fn glwe_tensor_square_apply_default<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        GLWETensor<R>: GLWEToBackendMut<BE>,
        GLWE<A>: crate::layouts::GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_square_apply(module, cnv_offset, res, a, a_effective_k, scratch)
    }

    fn glwe_tensor_relinearize_default<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: Data,
        A: Data,
        B: Data,
        GLWE<R>: GLWEToBackendMut<BE>,
        GLWETensorKeyPrepared<B, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
        GLWETensor<A>: crate::layouts::GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_relinearize(module, res, a, tsk, tsk_size, scratch)
    }

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_relinearize_tmp_bytes(module, res, a, tsk)
    }

    fn glwe_rotate_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate_tmp_bytes(module)
    }

    fn glwe_rotate_default<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GLWEBackendMut<'r, BE>, a: &GLWEBackendRef<'a, BE>) {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate(module, k, res, a)
    }

    fn glwe_rotate_assign_default<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GLWEBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate_assign(module, k, res, scratch)
    }

    fn ggsw_rotate_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_tmp_bytes_default(module)
    }

    fn ggsw_rotate_default<'r, 'a>(module: &Module<BE>, k: i64, res: &mut GGSWBackendMut<'r, BE>, a: &GGSWBackendRef<'a, BE>) {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_default(module, k, res, a)
    }

    fn ggsw_rotate_assign_default<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut GGSWBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        ScratchArena<'s, BE>: poulpy_hal::api::ScratchAvailable,
    {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_assign_default(module, k, res, scratch)
    }

    fn glwe_mul_xp_minus_one_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEMulXpMinusOneDefault<BE>>::glwe_mul_xp_minus_one(module, k, res, a)
    }

    fn glwe_mul_xp_minus_one_assign_default<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        <Module<BE> as GLWEMulXpMinusOneDefault<BE>>::glwe_mul_xp_minus_one_assign(module, k, res, scratch)
    }

    fn glwe_shift_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes(module)
    }

    fn glwe_rsh_default<'s, R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_rsh(module, k, res, scratch)
    }

    fn glwe_lsh_assign_default<'s, R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendMut<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_assign(module, res, k, scratch)
    }

    fn glwe_lsh_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh(module, res, a, k, scratch)
    }

    fn glwe_lsh_add_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_add(module, res, a, k, scratch)
    }

    fn glwe_lsh_sub_default<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_sub(module, res, a, k, scratch)
    }

    fn glwe_normalize_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes(module)
    }

    fn glwe_maybe_cross_normalize_to_ref_default<'a>(
        module: &Module<BE>,
        glwe: &'a GLWEBackendRef<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendRef<'a, BE> {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_maybe_cross_normalize_to_ref(
            module,
            glwe,
            target_base2k,
            tmp_slot,
            scratch,
        )
    }

    fn glwe_maybe_cross_normalize_to_mut_default<'a>(
        module: &Module<BE>,
        glwe: &'a mut GLWEBackendMut<'a, BE>,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWEBackendMut<'a, BE>>,
        scratch: &'a mut ScratchArena<'a, BE>,
    ) -> GLWEBackendMut<'a, BE> {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_maybe_cross_normalize_to_mut(
            module,
            glwe,
            target_base2k,
            tmp_slot,
            scratch,
        )
    }

    fn glwe_normalize_default<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize(module, res, a, scratch)
    }

    fn glwe_normalize_assign_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize_assign(module, res, scratch)
    }

    fn glwe_trace_galois_elements_default(module: &Module<BE>) -> Vec<i64> {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_galois_elements_default(module)
    }

    fn glwe_trace_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn glwe_trace_default<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    {
        let mut scratch_local = scratch.borrow();
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_default(module, res, skip, a, keys, &mut scratch_local)
    }

    fn glwe_trace_assign_default<'s, R, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
    {
        let mut scratch_local = scratch.borrow();
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_assign_default(module, res, skip, keys, &mut scratch_local)
    }

    fn glwe_pack_galois_elements_default(module: &Module<BE>) -> Vec<i64> {
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_galois_elements_default(module)
    }

    fn glwe_pack_tmp_bytes_default<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_tmp_bytes_default(module, res, key)
    }

    fn glwe_pack_default<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: crate::layouts::GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
        for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
        GLWE<Vec<u8>>: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEToBackendRef<BE>,
    {
        let mut scratch_local = scratch.borrow();
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_default(module, res, a, log_gap_out, keys, &mut scratch_local)
    }

    fn packer_add_default<'s, A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker<BE::OwnedBuf>,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: Backend,
    {
        <Module<BE> as GLWEPackerOpsDefault<BE>>::packer_add_default(module, packer, a, i, auto_keys, scratch)
    }
}

/// Delegate the `GLWERotateImpl` family to another host backend through the
/// module-owned transfer API.
///
/// This intentionally routes values through `upload_glwe` / `download_glwe`
/// so a partial backend can keep explicit ownership boundaries when it falls
/// back to another backend's implementation.
#[macro_export]
macro_rules! impl_glwe_rotate_impl_from {
    ($be:ty, $from:ty) => {
        unsafe impl $crate::oep::GLWERotateImpl<$be> for $be {
            fn glwe_rotate_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);
                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_tmp_bytes(&delegate)
            }

            fn glwe_rotate<R, A>(module: &poulpy_hal::layouts::Module<$be>, k: i64, res: &mut R, a: &A)
            where
                R: $crate::layouts::GLWEToBackendMut,
                A: $crate::layouts::GLWEToBackendRef,
            {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);

                let a_host: $crate::layouts::GLWE<Vec<u8>> =
                    poulpy_hal::layouts::ToOwnedDeep::to_owned_deep(&$crate::layouts::GLWEToBackendRef::to_backend_ref(a));
                let a_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> = a_host.reinterpret::<$be>();

                let res_infos = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                let res_host: $crate::layouts::GLWE<Vec<u8>> = delegate.glwe_alloc_from_infos(&res_infos);
                let res_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    res_host.reinterpret::<$be>();

                let a_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &a_src);
                let mut res_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &res_src);

                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate(
                    &delegate,
                    k,
                    &mut res_delegate,
                    &a_delegate,
                );

                let res_back: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    $crate::api::ModuleTransfer::download_glwe::<$from>(&delegate, &res_delegate);
                let res_back_ref = $crate::layouts::GLWEToBackendRef::to_backend_ref(&res_back);

                let mut bytes = Vec::new();
                poulpy_hal::layouts::WriterTo::write_to(&res_back_ref, &mut bytes)
                    .expect("failed to serialize delegated GLWE rotate result");

                let mut cursor = std::io::Cursor::new(bytes);
                let mut res_mut = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                poulpy_hal::layouts::ReaderFrom::read_from(&mut res_mut, &mut cursor)
                    .expect("failed to write delegated GLWE rotate result back");
            }

            fn glwe_rotate_assign<'s, R>(
                module: &poulpy_hal::layouts::Module<$be>,
                k: i64,
                res: &mut R,
                _scratch: &mut poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut,
            {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);

                let res_host: $crate::layouts::GLWE<Vec<u8>> =
                    poulpy_hal::layouts::ToOwnedDeep::to_owned_deep(&$crate::layouts::GLWEToBackendMut::to_backend_mut(res));
                let res_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    res_host.reinterpret::<$be>();
                let mut res_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &res_src);

                let mut scratch_owned: poulpy_hal::layouts::ScratchOwned<$from> =
                    <poulpy_hal::layouts::ScratchOwned<$from> as poulpy_hal::api::ScratchOwnedAlloc<$from>>::alloc(
                        <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_tmp_bytes(&delegate),
                    );
                let scratch_delegate =
                    <poulpy_hal::layouts::ScratchOwned<$from> as poulpy_hal::api::ScratchOwnedBorrow<$from>>::borrow(
                        &mut scratch_owned,
                    );

                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_assign(
                    &delegate,
                    k,
                    &mut res_delegate,
                    scratch_delegate,
                );

                let res_back: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    $crate::api::ModuleTransfer::download_glwe::<$from>(&delegate, &res_delegate);
                let res_back_ref = $crate::layouts::GLWEToBackendRef::to_backend_ref(&res_back);

                let mut bytes = Vec::new();
                poulpy_hal::layouts::WriterTo::write_to(&res_back_ref, &mut bytes)
                    .expect("failed to serialize delegated GLWE rotate inplace result");

                let mut cursor = std::io::Cursor::new(bytes);
                let mut res_mut = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                poulpy_hal::layouts::ReaderFrom::read_from(&mut res_mut, &mut cursor)
                    .expect("failed to write delegated GLWE rotate inplace result back");
            }
        }
    };
}
