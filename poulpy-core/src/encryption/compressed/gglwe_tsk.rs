use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, SvpPPolBytesOf, SvpPrepare, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWECompressedEncryptSk, GetDistribution, ScratchTakeCore,
    encryption::gglwe_tsk::TensorKeyEncryptSk,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPrepared, GLWESecretPreparedAlloc, GLWESecretToRef, LWEInfos, Rank,
        TensorKeyCompressedAtMut, compressed::TensorKeyCompressed,
    },
};

impl TensorKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWETensorKeyCompressedEncryptSk<BE>,
    {
        module.tensor_key_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> TensorKeyCompressed<DataSelf> {
    pub fn encrypt_sk<S, M, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretToRef + GetDistribution,
        M: GGLWETensorKeyCompressedEncryptSk<BE>,
    {
        module.tensor_key_compressed_encrypt_sk(self, sk, seed_xa, source_xe, scratch);
    }
}

pub trait GGLWETensorKeyCompressedEncryptSk<BE: Backend> {
    fn tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn tensor_key_compressed_encrypt_sk<R, S, D>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        R: TensorKeyCompressedAtMut<D> + GGLWEInfos,
        S: GLWESecretToRef + GetDistribution;
}

impl<BE: Backend> GGLWETensorKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWECompressedEncryptSk<BE>
        + TensorKeyEncryptSk<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + SvpPrepare<BE>
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + GLWESecretPreparedAlloc<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchTakeCore<BE>,
{
    fn tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GLWESecretPrepared::bytes_of(self, infos.rank_out())
            + self.bytes_of_vec_znx_dft(infos.rank_out().into(), 1)
            + self.bytes_of_vec_znx_big(1, 1)
            + self.bytes_of_vec_znx_dft(1, 1)
            + GLWESecret::bytes_of(self.n().into(), Rank(1))
            + self.gglwe_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn tensor_key_compressed_encrypt_sk<R, S, D>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        R: GGLWEInfos + TensorKeyCompressedAtMut<D>,
        S: GLWESecretToRef + GetDistribution,
    {
        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(self, res.rank_out());
        sk_dft_prep.prepare(self, sk);

        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank_out(), sk.rank());
            assert_eq!(res.n(), sk.n());
        }

        // let n: usize = sk.n().into();
        let rank: usize = res.rank_out().into();

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, rank, 1);

        for i in 0..rank {
            self.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        }

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(self, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(self, Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(self, 1, 1);

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

                self.gglwe_compressed_encrypt_sk(
                    res.at_mut(i, j),
                    &sk_ij.data,
                    &sk_dft_prep,
                    seed_xa_tmp,
                    source_xe,
                    scratch_5,
                );
            }
        }
    }
}
