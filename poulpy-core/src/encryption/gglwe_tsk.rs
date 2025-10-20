use poulpy_hal::{
    api::{
        ModuleN, ScratchTakeBasic, SvpApplyDftToDft, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    GGLWEEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, LWEInfos, Rank, TensorKey, TensorKeyToMut,
        prepared::{GLWESecretPrepare, GLWESecretPrepared, GLWESecretPreparedAlloc},
    },
};

impl TensorKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: TensorKeyEncryptSk<BE>,
    {
        module.tensor_key_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> TensorKey<DataSelf> {
    pub fn encrypt_sk<M, S, BE: Backend>(
        &mut self,
        module: &M,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: TensorKeyEncryptSk<BE>,
        S: GLWESecretToRef + GetDistribution,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.tensor_key_encrypt_sk(self, sk, source_xa, source_xe, scratch);
    }
}

pub trait TensorKeyEncryptSk<BE: Backend> {
    fn tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: TensorKeyToMut,
        S: GLWESecretToRef + GetDistribution;
}

impl<BE: Backend> TensorKeyEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + GLWESecretPreparedAlloc<BE>
        + GLWESecretPrepare<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GLWESecretPrepared::bytes_of(self, infos.rank_out())
            + self.bytes_of_vec_znx_dft(infos.rank_out().into(), 1)
            + self.bytes_of_vec_znx_big(1, 1)
            + self.bytes_of_vec_znx_dft(1, 1)
            + GLWESecret::bytes_of(self.n().into(), Rank(1))
            + GGLWE::encrypt_sk_tmp_bytes(self, infos)
    }

    fn tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: TensorKeyToMut,
        S: GLWESecretToRef + GetDistribution,
    {
        let res: &mut TensorKey<&mut [u8]> = &mut res.to_mut();

        // let n: RingDegree = sk.n();
        let rank: Rank = res.rank_out();

        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared(self, rank);
        sk_prepared.prepare(self, sk);

        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, rank.into(), 1);

        (0..rank.into()).for_each(|i| {
            self.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(self, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(self, Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(self, 1, 1);

        (0..rank.into()).for_each(|i| {
            (i..rank.into()).for_each(|j| {
                self.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_prepared.data, j, &sk_dft, i);

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

                res.at_mut(i, j).encrypt_sk(
                    self,
                    &sk_ij.data,
                    &sk_prepared,
                    source_xa,
                    source_xe,
                    scratch_5,
                );
            });
        })
    }
}
