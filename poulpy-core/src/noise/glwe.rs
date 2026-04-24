use poulpy_hal::layouts::{Backend, HostBackend, Module, ScratchArena, Stats};

use crate::{
    GLWENormalize, GLWESub, ScratchArenaTakeCore,
    api::GLWENoise,
    decryption::{GLWEDecrypt, GLWEDecryptDefault, glwe_decrypt_backend_inner},
    layouts::{
        GLWE, GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWEToBackendRef, GLWEToRef, LWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub(crate) fn glwe_noise_backend_inner<'s, M, P, BE: Backend>(
    module: &M,
    res_ref: &GLWE<&[u8]>,
    res_backend: &GLWEBackendRef<'_, BE>,
    pt_want: &P,
    sk_backend: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) -> Stats
where
    M: GLWENoise<BE> + GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWESub + GLWENormalize<BE>,
    P: GLWEToRef,
    BE: HostBackend,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    assert!(
        scratch.available() >= module.glwe_noise_tmp_bytes(res_ref),
        "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
        scratch.available(),
        module.glwe_noise_tmp_bytes(res_ref)
    );

    let (mut pt_have, mut scratch_1) = scratch.borrow().take_glwe_plaintext(res_ref);
    glwe_decrypt_backend_inner(module, res_ref, res_backend, &mut pt_have, sk_backend, &mut scratch_1);
    module.glwe_sub_inplace(&mut pt_have, pt_want);
    let pt_base2k = pt_have.base2k();
    let mut pt_have_backend = GLWE {
        base2k: pt_have.base2k,
        data: pt_have.data,
    };
    module.glwe_normalize_inplace(&mut pt_have_backend, &mut scratch_1);
    pt_have_backend.data.stats(pt_base2k.into(), 0)
}

impl<BE: Backend + HostBackend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWESub + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_normalize_tmp_bytes().max(self.glwe_decrypt_tmp_bytes(infos));

        lvl_0 + lvl_1
    }

    fn glwe_noise<'s, R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'s, BE>) -> Stats
    where
        R: GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        let res_ref = res.to_ref();
        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        glwe_noise_backend_inner(self, &res_ref, &res_backend, pt_want, &sk_backend, scratch)
    }
}
