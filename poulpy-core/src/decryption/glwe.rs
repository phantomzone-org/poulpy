use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDftInplace, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, VecZnxBigReborrowBackendRef},
};

pub use crate::api::GLWEDecrypt;
use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GLWE, GLWEBackendRef, GLWEInfos, GLWEPlaintextBackendMut, GLWEPlaintextToBackendMut, GLWEPlaintextToMut,
        GLWEToBackendRef, GLWEToRef, LWEInfos, SetLWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub(crate) trait GLWEDecryptDefault<BE: Backend>:
    Sized
    + ModuleN
    + VecZnxDftBytesOf
    + VecZnxNormalizeTmpBytes
    + VecZnxBigBytesOf
    + VecZnxBigFromSmallBackend<BE>
    + VecZnxDftApply<BE>
    + SvpApplyDftToDftAssign<BE>
    + VecZnxIdftApplyConsume<BE>
    + VecZnxBigAddAssign<BE>
    + VecZnxBigNormalize<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.bytes_of_vec_znx_big(1, size);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(1, size).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    fn glwe_decrypt_default<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend + 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_ref: &GLWE<&[u8]> = &res.to_ref();
        let res_backend = res.to_backend_ref();
        let mut pt_backend = pt.to_backend_mut();
        let sk_backend = sk.to_backend_ref();

        glwe_decrypt_backend_inner(self, res_ref, &res_backend, &mut pt_backend, &sk_backend, scratch);
    }
}

impl<BE: Backend> GLWEDecryptDefault<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}

pub(crate) fn glwe_decrypt_backend_inner<'s, M, BE: Backend + HostBackend + 's>(
    module: &M,
    res_ref: &GLWE<&[u8]>,
    res: &GLWEBackendRef<'_, BE>,
    pt: &mut GLWEPlaintextBackendMut<'_, BE>,
    sk: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: GLWEDecryptDefault<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res_ref.rank(), sk.rank());
        assert_eq!(res_ref.n(), sk.n());
        assert_eq!(pt.n(), sk.n());
    }
    assert!(
        scratch.available() >= module.glwe_decrypt_tmp_bytes_default(res_ref),
        "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
        scratch.available(),
        module.glwe_decrypt_tmp_bytes_default(res_ref)
    );

    let cols: usize = (res.rank() + 1).into();
    let (mut c0_big, mut scratch_1) = scratch.borrow().take_vec_znx_big(module, 1, res.size());
    module.vec_znx_big_from_small_backend(&mut c0_big, 0, &res.data, 0);

    for i in 1..cols {
        let (mut ci_dft, _) = scratch_1.borrow().take_vec_znx_dft(module, 1, res.size());
        module.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &res.data, i);
        module.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
        let ci_big = module.vec_znx_idft_apply_consume(ci_dft);
        let ci_big_ref = ci_big.reborrow_backend_ref();
        module.vec_znx_big_add_assign(&mut c0_big, 0, &ci_big_ref, 0);
    }

    let c0_big_ref = c0_big.reborrow_backend_ref();
    let pt_base2k = pt.base2k();
    let _ = scratch_1.apply_mut(|scratch| {
        module.vec_znx_big_normalize(
            &mut pt.data,
            pt_base2k.into(),
            0,
            0,
            &c0_big_ref,
            res.base2k().into(),
            0,
            scratch,
        )
    });
}
