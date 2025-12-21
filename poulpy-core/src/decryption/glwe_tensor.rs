use poulpy_hal::{
    api::{ScratchTakeBasic, SvpPPolBytesOf},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxView, ZnxViewMut},
};

use crate::{
    GLWEDecrypt, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensor, GLWESecretTensorPrepared, GLWETensor},
};

impl GLWETensor<Vec<u8>> {
    pub fn decrypt_tmp_bytes<A, M, BE: Backend>(module: &M, a_infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWETensorDecrypt<BE>,
    {
        module.glwe_tensor_decrypt_tmp_bytes(a_infos)
    }
}

impl<DataSelf: DataRef> GLWETensor<DataSelf> {
    pub fn decrypt<P, S0, S1, M, BE: Backend>(
        &self,
        module: &M,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        P: DataMut,
        S0: DataRef,
        S1: DataRef,
        M: GLWETensorDecrypt<BE>,
        Scratch<BE>: ScratchTakeBasic,
    {
        module.glwe_tensor_decrypt(self, pt, sk, sk_tensor, scratch);
    }
}

pub trait GLWETensorDecrypt<BE: Backend> {
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;
    fn glwe_tensor_decrypt<R, P, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef;
}

impl<BE: Backend> GLWETensorDecrypt<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE> + SvpPPolBytesOf,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_decrypt_tmp_bytes(infos)
    }

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef,
    {
        let rank: usize = sk.rank().as_usize();

        let (mut sk_grouped, scratch_1) = scratch.take_glwe_secret_prepared(self, (GLWESecretTensor::pairs(rank) + rank).into());

        for i in 0..rank {
            sk_grouped.data.at_mut(i, 0).copy_from_slice(sk.data.at(i, 0));
        }

        for i in 0..sk_grouped.rank().as_usize() - rank {
            sk_grouped.data.at_mut(i + rank, 0).copy_from_slice(sk_tensor.data.at(i, 0));
        }

        self.glwe_decrypt(res, pt, &sk_grouped, scratch_1);
    }
}
