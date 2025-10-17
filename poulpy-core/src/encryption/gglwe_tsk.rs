use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    oep::{VecZnxAddScalarInplaceImpl, VecZnxBigAllocBytesImpl, VecZnxDftApplyImpl, SvpApplyDftToDftImpl, VecZnxIdftApplyTmpAImpl, VecZnxBigNormalizeImpl},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GetDist, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey, LWEInfos, Rank, TensorKey, TensorKeyToMut,
        prepared::GLWESecretPrepared,
    },
    encryption::gglwe_ksk::GLWESwitchingKeyEncryptSk,
};

impl TensorKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWETensorKeyEncryptSk<BE>
    {
        module.gglwe_tensor_key_encrypt_sk_tmp_bytes(infos)
    }
    // pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    // where
    //     A: GGLWEInfos,
    //     Module<B>: ModuleN + SvpPPolBytesOf + SvpPPolAlloc<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes + VecZnxBigBytesOf,
    // {
    //     GLWESecretPrepared::bytes_of(module, infos.rank_out())
    //         + module.bytes_of_vec_znx_dft(infos.rank_out().into(), 1)
    //         + module.bytes_of_vec_znx_big(1, 1)
    //         + module.bytes_of_vec_znx_dft(1, 1)
    //         + GLWESecret::bytes_of(module, Rank(1))
    //         + GLWESwitchingKey::encrypt_sk_tmp_bytes(module, infos)
    // }
}

impl<DataSelf: DataMut> TensorKey<DataSelf> {
    pub fn encrypt_sk<M, DataSk: DataRef, BE: Backend>(
        &mut self,
        module: &M,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GGLWETensorKeyEncryptSk<BE>,
        GLWESecret<DataSk>: GetDist,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.gglwe_tensor_key_encrypt_sk(self, sk, source_xa, source_xe, scratch);
    }

    // pub fn encrypt_sk<DataSk: DataRef, B: Backend>(
    //     &mut self,
    //     module: &Module<B>,
    //     sk: &GLWESecret<DataSk>,
    //     source_xa: &mut Source,
    //     source_xe: &mut Source,
    //     scratch: &mut Scratch<B>,
    // ) where
    //     GLWESecret<DataSk>: GetDist,
    //     Module<B>: ModuleN
    //         + SvpApplyDftToDft<B>
    //         + VecZnxIdftApplyTmpA<B>
    //         + VecZnxAddScalarInplace
    //         + VecZnxDftBytesOf
    //         + VecZnxBigNormalize<B>
    //         + VecZnxDftApply<B>
    //         + SvpApplyDftToDftInplace<B>
    //         + VecZnxIdftApplyConsume<B>
    //         + VecZnxNormalizeTmpBytes
    //         + VecZnxFillUniform
    //         + VecZnxSubInplace
    //         + VecZnxAddInplace
    //         + VecZnxNormalizeInplace<B>
    //         + VecZnxAddNormal
    //         + VecZnxNormalize<B>
    //         + VecZnxSub
    //         + SvpPrepare<B>
    //         + VecZnxSwitchRing
    //         + SvpPPolBytesOf
    //         + VecZnxBigAllocBytesImpl<B>
    //         + VecZnxBigBytesOf
    //         + SvpPPolAlloc<B>,
    //     Scratch<B>: ScratchTakeBasic + ScratchTakeCore<B>,
    // {
    //     #[cfg(debug_assertions)]
    //     {
    //         assert_eq!(self.rank_out(), sk.rank());
    //         assert_eq!(self.n(), sk.n());
    //     }

    //     // let n: RingDegree = sk.n();
    //     let rank: Rank = self.rank_out();

    //     let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(module, rank);
    //     sk_dft_prep.prepare(module, sk);

    //     let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(module, rank.into(), 1);

    //     (0..rank.into()).for_each(|i| {
    //         module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
    //     });

    //     let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(module, 1, 1);
    //     let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(module, Rank(1));
    //     let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(module, 1, 1);

    //     (0..rank.into()).for_each(|i| {
    //         (i..rank.into()).for_each(|j| {
    //             module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_dft_prep.data, j, &sk_dft, i);

    //             module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
    //             module.vec_znx_big_normalize(
    //                 self.base2k().into(),
    //                 &mut sk_ij.data.as_vec_znx_mut(),
    //                 0,
    //                 self.base2k().into(),
    //                 &sk_ij_big,
    //                 0,
    //                 scratch_5,
    //             );

    //             self.at_mut(i, j)
    //                 .encrypt_sk(module, &sk_ij, sk, source_xa, source_xe, scratch_5);
    //         });
    //     })
    // }
}

pub trait GGLWETensorKeyEncryptSk<BE: Backend> 
where
    Self: Sized
        + ModuleN
        + SvpPPolBytesOf
        + SvpPPolAlloc<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf,
{
    fn gglwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: TensorKeyToMut,
        S: GLWESecretToRef + GetDist;
}

impl<BE: Backend> GGLWETensorKeyEncryptSk<BE> for Module<BE> where
    Module<BE>: ModuleN
        + SvpPPolBytesOf
        + SvpPPolAlloc<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxAddScalarInplaceImpl<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftApplyImpl<BE>
        + SvpApplyDftToDftImpl<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpAImpl<BE>
        + VecZnxBigNormalizeImpl<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxAddScalarInplaceImpl<BE>
        + SvpPrepare<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchTakeCore<BE>,
{
    fn gglwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GLWESecretPrepared::bytes_of(self, infos.rank_out())
        + self.bytes_of_vec_znx_dft(infos.rank_out().into(), 1)
        + self.bytes_of_vec_znx_big(1, 1)
        + self.bytes_of_vec_znx_dft(1, 1)
        + GLWESecret::bytes_of(self, Rank(1))
        + GLWESwitchingKey::encrypt_sk_tmp_bytes(self, infos)        
    }

    fn gglwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: TensorKeyToMut,
        S: GLWESecretToRef + GetDist,
    {
        let res: &mut TensorKey<&mut [u8]> = &mut res.to_mut();

        // let n: RingDegree = sk.n();
        let rank: Rank = res.rank_out();

        let (mut sk_dft_prep, scratch_1) = scratch.take_glwe_secret_prepared(self, rank);
        sk_dft_prep.prepare(self, sk);

        let sk: &GLWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank_out(), sk.rank());
            assert_eq!(res.n(), sk.n());
        }

        let (mut sk_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, rank.into(), 1);

        (0..rank.into()).for_each(|i| {
            self.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        });

        let (mut sk_ij_big, scratch_3) = scratch_2.take_vec_znx_big(self, 1, 1);
        let (mut sk_ij, scratch_4) = scratch_3.take_glwe_secret(self, Rank(1));
        let (mut sk_ij_dft, scratch_5) = scratch_4.take_vec_znx_dft(self, 1, 1);

        (0..rank.into()).for_each(|i| {
            (i..rank.into()).for_each(|j| {
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

                res.at_mut(i, j)
                    .encrypt_sk(self, &sk_ij, sk, source_xa, source_xe, scratch_5);
            });
        })        
    }
}   