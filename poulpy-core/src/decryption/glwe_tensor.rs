use poulpy_hal::{
    api::SvpPPolBytesOf,
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, ZnxView, ZnxViewMut},
};

pub use crate::api::GLWETensorDecrypt;
use crate::{
    ScratchArenaTakeCore,
    decryption::GLWEDecryptDefault,
    decryption::glwe::glwe_decrypt_backend_inner,
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWESecretPrepared, GLWESecretTensor, GLWESecretTensorPrepared,
        GLWETensor, GLWEToBackendRef,
        prepared::{GLWESecretPreparedFactory, glwe_secret_prepared_backend_ref_from_mut},
    },
};

pub(crate) trait GLWETensorDecryptDefault<BE: Backend>:
    Sized + GLWEDecryptDefault<BE> + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_tensor_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let rank: usize = infos.rank().into();
        let lvl_0: usize = self.glwe_secret_prepared_bytes_of((GLWESecretTensor::pairs(rank) + rank).into());
        let lvl_1: usize = self.glwe_decrypt_tmp_bytes_default(infos);

        lvl_0 + lvl_1
    }

    fn glwe_tensor_decrypt_default<R, P, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: HostDataRef,
        GLWETensor<R>: GLWEToBackendRef<BE> + GLWEInfos,
        P: HostDataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + crate::layouts::SetLWEInfos,
        S0: HostDataRef,
        S1: HostDataRef,
        BE: HostBackend,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_decrypt_tmp_bytes_default(res),
            "scratch.available(): {} < GLWETensorDecrypt::glwe_tensor_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_decrypt_tmp_bytes_default(res)
        );

        let rank: usize = sk.rank().as_usize();

        let (mut sk_grouped, mut scratch_1) = scratch
            .borrow()
            .take_glwe_secret_prepared(self, (GLWESecretTensor::pairs(rank) + rank).into());

        for i in 0..rank {
            sk_grouped.data.at_mut(i, 0).copy_from_slice(sk.data.at(i, 0));
        }

        for i in 0..sk_grouped.rank().as_usize() - rank {
            sk_grouped.data.at_mut(i + rank, 0).copy_from_slice(sk_tensor.data.at(i, 0));
        }

        let res_backend = res.to_backend_ref();
        let mut pt_backend = pt.to_backend_mut();
        let sk_grouped_ref = glwe_secret_prepared_backend_ref_from_mut(&sk_grouped);
        glwe_decrypt_backend_inner(self, &res_backend, &mut pt_backend, &sk_grouped_ref, &mut scratch_1);
    }
}

impl<BE: Backend> GLWETensorDecryptDefault<BE> for Module<BE>
where
    Self: GLWEDecryptDefault<BE> + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
