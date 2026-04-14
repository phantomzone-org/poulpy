use poulpy_hal::{
    api::{ScratchAvailable, SvpPPolBytesOf},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
};

pub use crate::api::GLWETensorDecrypt;
use crate::{
    ScratchTakeCore,
    decryption::GLWEDecryptDefault,
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensor, GLWESecretTensorPrepared, GLWETensor,
        prepared::GLWESecretPreparedFactory,
    },
};

pub(crate) trait GLWETensorDecryptDefault<BE: Backend>:
    Sized + GLWEDecryptDefault<BE> + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>
where
    Scratch<BE>: ScratchTakeCore<BE>,
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

    fn glwe_tensor_decrypt_default<R, P, PM, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P, PM>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef,
    {
        assert!(
            scratch.available() >= self.glwe_tensor_decrypt_tmp_bytes_default(res),
            "scratch.available(): {} < GLWETensorDecrypt::glwe_tensor_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_decrypt_tmp_bytes_default(res)
        );

        let rank: usize = sk.rank().as_usize();

        let (mut sk_grouped, scratch_1) = scratch.take_glwe_secret_prepared(self, (GLWESecretTensor::pairs(rank) + rank).into());

        for i in 0..rank {
            sk_grouped.data.at_mut(i, 0).copy_from_slice(sk.data.at(i, 0));
        }

        for i in 0..sk_grouped.rank().as_usize() - rank {
            sk_grouped.data.at_mut(i + rank, 0).copy_from_slice(sk_tensor.data.at(i, 0));
        }

        self.glwe_decrypt_default(res, pt, &sk_grouped, scratch_1);
    }
}

impl<BE: Backend> GLWETensorDecryptDefault<BE> for Module<BE>
where
    Self: GLWEDecryptDefault<BE> + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
