use poulpy_hal::layouts::{
    Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, Stats, vec_znx_host_backend_ref,
};

use crate::vec_znx_host_ops::vec_znx_sub_assign;
use crate::{
    GLWENormalize, ScratchArenaTakeCore,
    api::GLWENoise,
    decryption::{GLWEDecrypt, GLWEDecryptDefault, glwe_decrypt_backend_inner},
    layouts::{
        GLWE, GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWEToBackendRef, LWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub(crate) fn glwe_noise_backend_inner<'s, M, BE>(
    module: &M,
    res_ref: &GLWE<&[u8]>,
    res_backend: &GLWEBackendRef<'_, BE>,
    pt_want_ref: &GLWE<&[u8]>,
    sk_backend: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) -> Stats
where
    M: GLWENoise<BE> + GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWENormalize<BE>,
    BE: HostBackend,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert!(
        scratch.available() >= module.glwe_noise_tmp_bytes(res_ref),
        "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
        scratch.available(),
        module.glwe_noise_tmp_bytes(res_ref)
    );

    let (mut pt_have, mut scratch_1) = scratch.borrow().take_glwe_plaintext(res_ref);
    glwe_decrypt_backend_inner(module, res_backend, &mut pt_have, sk_backend, &mut scratch_1);
    vec_znx_sub_assign(&mut pt_have.data, 0, &pt_want_ref.data, 0);
    let pt_base2k = pt_have.base2k();
    let mut pt_have_backend = GLWE {
        base2k: pt_have.base2k,
        data: pt_have.data,
    };
    module.glwe_normalize_assign(&mut pt_have_backend, &mut scratch_1);
    pt_have_backend.data.stats(pt_base2k.into(), 0)
}

impl<BE: Backend + HostBackend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
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
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_backend = res.to_backend_ref();
        let res_ref = GLWE {
            base2k: res_backend.base2k,
            data: vec_znx_host_backend_ref(&res_backend.data),
        };
        let pt_want_backend = pt_want.to_backend_ref();
        let pt_want_ref = GLWE {
            base2k: pt_want_backend.base2k,
            data: vec_znx_host_backend_ref(&pt_want_backend.data),
        };
        let sk_backend = sk_prepared.to_backend_ref();
        glwe_noise_backend_inner(self, &res_ref, &res_backend, &pt_want_ref, &sk_backend, scratch)
    }
}
