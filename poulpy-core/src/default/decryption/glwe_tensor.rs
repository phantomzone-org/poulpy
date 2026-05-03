use poulpy_hal::{
    api::{SvpPPolBytesOf, SvpPPolCopyBackend},
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    decryption::GLWEDecryptDefault,
    decryption::glwe::glwe_decrypt_backend_inner,
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensor, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
        GLWEToBackendRef,
        prepared::{
            GLWESecretPreparedFactory, GLWESecretPreparedToBackendMut, GLWESecretPreparedToBackendRef,
            GLWESecretTensorPreparedToBackendRef, glwe_secret_prepared_backend_ref_from_mut,
        },
    },
};

pub(crate) trait GLWETensorDecryptDefault<BE: Backend>:
    Sized + GLWEDecryptDefault<BE> + SvpPPolBytesOf + SvpPPolCopyBackend<BE> + GLWESecretPreparedFactory<BE>
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

    fn glwe_tensor_decrypt_default<R: Data, P: Data, S0: Data, S1: Data>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P>: GLWEToBackendMut<BE> + GLWEInfos + crate::layouts::SetLWEInfos,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos,
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

        {
            let mut binding = &mut sk_grouped;
            let mut grouped_backend = binding.to_backend_mut();
            let sk_backend = sk.to_backend_ref();
            let sk_tensor_backend = sk_tensor.to_backend_ref();

            for i in 0..rank {
                self.svp_ppol_copy_backend(&mut grouped_backend.data, i, &sk_backend.data, i);
            }

            for i in 0..(grouped_backend.rank().as_usize() - rank) {
                self.svp_ppol_copy_backend(&mut grouped_backend.data, i + rank, &sk_tensor_backend.data, i);
            }
        }

        let res_backend = res.to_backend_ref();
        let mut pt_backend = pt.to_backend_mut();
        let sk_grouped_ref = glwe_secret_prepared_backend_ref_from_mut(&sk_grouped);
        glwe_decrypt_backend_inner(self, &res_backend, &mut pt_backend, &sk_grouped_ref, &mut scratch_1);
    }
}

impl<BE: Backend> GLWETensorDecryptDefault<BE> for Module<BE>
where
    Self: GLWEDecryptDefault<BE> + SvpPPolBytesOf + SvpPPolCopyBackend<BE> + GLWESecretPreparedFactory<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
}
