use poulpy_hal::{
    api::{
        SvpApplyDftToDft, SvpPPolBytesOf, SvpPrepare, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    encryption::compressed::gglwe_ksk::GGLWEKeyCompressedEncryptSk,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos, Rank, TensorKey,
        compressed::{TensorKeyCompressed, TensorKeyCompressedToMut},
    },
};

impl TensorKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: SvpPPolBytesOf + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigBytesOf,
    {
        TensorKey::encrypt_sk_scratch_space(module, infos)
    }
}

pub trait GGLWETensorKeyCompressedEncryptSk<B: Backend> {
    fn gglwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: TensorKeyCompressedToMut,
        S: GLWESecretToRef;
}

impl<B: Backend> GGLWETensorKeyCompressedEncryptSk<B> for Module<B>
where
    Module<B>: GGLWEKeyCompressedEncryptSk<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDft<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxBigNormalize<B>
        + SvpPrepare<B>,
    Scratch<B>:,
{
    fn gglwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: TensorKeyCompressedToMut,
        S: GLWESecretToRef,
    {
        let res: &mut TensorKeyCompressed<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank_out(), sk.rank());
            assert_eq!(res.n(), sk.n());
        }

        let n: usize = sk.n().into();
        let rank: usize = res.rank_out().into();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(sk.n(), res.rank_out());
        sk_dft_prep.prepare(self, sk, scratch_1);

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(n, rank, 1);

        for i in 0..rank {
            self.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        }

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(n, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(sk.n(), Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(n, 1, 1);

        let mut source_xa: Source = Source::new(seed_xa);

        for i in 0..rank {
            for j in i..rank {
                self.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

                self.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                self.vec_znx_big_normalize(
                    res.base2k().into(),
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
                    res.base2k().into(),
                    &sk_ij_big,
                    0,
                    scratch_5,
                );

                let (seed_xa_tmp, _) = source_xa.branch();

                self.gglwe_key_compressed_encrypt_sk(
                    res.at_mut(i, j),
                    &sk_ij,
                    sk,
                    seed_xa_tmp,
                    source_xe,
                    scratch_5,
                );
            }
        }
    }
}

impl<DataSelf: DataMut> TensorKeyCompressed<DataSelf> {
    pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecret<DataSk>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWETensorKeyCompressedEncryptSk<B>,
    {
        module.gglwe_tensor_key_encrypt_sk(self, sk, seed_xa, source_xe, scratch);
    }
}
