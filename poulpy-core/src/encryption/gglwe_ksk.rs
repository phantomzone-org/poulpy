use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
        ScratchTakeBasic,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey, GLWESwitchingKeyToMut, LWEInfos, prepared::GLWESecretPrepared,
    },
    encryption::gglwe_ct::GGLWEEncryptSk,
};

impl GLWESwitchingKey<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyEncryptSk<BE>,
    {
        module.glwe_switching_key_encrypt_sk_tmp_bytes(infos)
    }

    pub fn encrypt_pk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyEncryptPk<BE>,
    {
        module.glwe_switching_key_encrypt_pk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GLWESwitchingKey<DataSelf> {

    pub fn encrypt_sk<M, DataSkIn: DataRef, DataSkOut: DataRef, BE: Backend>(
        &mut self,
        module: &M,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWESwitchingKeyEncryptSk<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.glwe_switching_key_encrypt_sk(self, sk_in, sk_out, source_xa, source_xe, scratch);
    }


    // #[allow(clippy::too_many_arguments)]
    // pub fn encrypt_sk<DataSkIn: DataRef, DataSkOut: DataRef, B: Backend>(
    //     &mut self,
    //     module: &Module<B>,
    //     sk_in: &GLWESecret<DataSkIn>,
    //     sk_out: &GLWESecret<DataSkOut>,
    //     source_xa: &mut Source,
    //     source_xe: &mut Source,
    //     scratch: &mut Scratch<B>,
    // ) where
    //     Module<B>: ModuleN
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
    //         + SvpPPolAlloc<B>,
    //     Scratch<B>: ScratchAvailable + ScratchTakeBasic + ScratchTakeCore<B>,
    // {
    //     #[cfg(debug_assertions)]
    //     {
    //         assert!(sk_in.n().0 <= module.n() as u32);
    //         assert!(sk_out.n().0 <= module.n() as u32);
    //         assert!(
    //             scratch.available() >= GLWESwitchingKey::encrypt_sk_tmp_bytes(module, self),
    //             "scratch.available()={} < GLWESwitchingKey::encrypt_sk_tmp_bytes={}",
    //             scratch.available(),
    //             GLWESwitchingKey::encrypt_sk_tmp_bytes(module, self)
    //         )
    //     }

    //     // let n: usize = sk_in.n().max(sk_out.n()).into();

    //     let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(module, sk_in.rank().into());
    //     (0..sk_in.rank().into()).for_each(|i| {
    //         module.vec_znx_switch_ring(
    //             &mut sk_in_tmp.as_vec_znx_mut(),
    //             i,
    //             &sk_in.data.as_vec_znx(),
    //             i,
    //         );
    //     });

    //     let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(module, sk_out.rank());
    //     {
    //         let (mut tmp, _) = scratch_2.take_scalar_znx(module, 1);
    //         (0..sk_out.rank().into()).for_each(|i| {
    //             module.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
    //             module.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
    //         });
    //     }

    //     self.key.encrypt_sk(
    //         module,
    //         &sk_in_tmp,
    //         &sk_out_tmp,
    //         source_xa,
    //         source_xe,
    //         scratch_2,
    //     );
    //     self.sk_in_n = sk_in.n().into();
    //     self.sk_out_n = sk_out.n().into();
    // }
}


pub trait GLWESwitchingKeyEncryptSk<BE: Backend> 
where
    Self: Sized
        + ModuleN
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + SvpPPolAlloc<BE>
        + GGLWEEncryptSk<BE>,
{
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, DataSkIn, DataSkOut>(
        &self,
        res: &mut R,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) 
    where 
        R: GLWESwitchingKeyToMut,
        DataSkIn: DataRef,
        DataSkOut: DataRef,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

}

impl<BE: Backend> GLWESwitchingKeyEncryptSk<BE> for Module<BE> where
    Self: ModuleN
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + SvpPPolAlloc<BE>
        + GGLWEEncryptSk<BE>
        + VecZnxSwitchRing
        + SvpPrepare<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddInplace
        + VecZnxAddNormal
        + VecZnxSub
{

    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        (GGLWE::encrypt_sk_tmp_bytes(self, infos) | ScalarZnx::bytes_of(self.n(), 1))
            + ScalarZnx::bytes_of(self.n(), infos.rank_in().into())
            + GLWESecretPrepared::bytes_of_from_infos(self, &infos.glwe_layout())
    }

    fn glwe_switching_key_encrypt_sk<R, DataSkIn, DataSkOut>(
        &self,
        res: &mut R,
        sk_in: &GLWESecret<DataSkIn>,
        sk_out: &GLWESecret<DataSkOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) 
    where 
        R: GLWESwitchingKeyToMut,
        DataSkIn: DataRef,
        DataSkOut: DataRef,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {

        let res: &mut GLWESwitchingKey<&mut [u8]> = &mut res.to_mut();
        let sk_in: &GLWESecret<&[u8]> = &sk_in.to_ref();
        let sk_out: &GLWESecret<&[u8]> = &sk_out.to_ref();

        #[cfg(debug_assertions)]
        {
            assert!(sk_in.n().0 <= self.n() as u32);
            assert!(sk_out.n().0 <= self.n() as u32);
            assert!(
                scratch.available() >= GLWESwitchingKey::encrypt_sk_tmp_bytes(self, res),
                "scratch.available()={} < GLWESwitchingKey::encrypt_sk_tmp_bytes={}",
                scratch.available(),
                GLWESwitchingKey::encrypt_sk_tmp_bytes(self, res)
            )
        }

        // let n: usize = sk_in.n().max(sk_out.n()).into();

        let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(self, sk_in.rank().into());
        (0..sk_in.rank().into()).for_each(|i| {
            self.vec_znx_switch_ring(
                &mut sk_in_tmp.as_vec_znx_mut(),
                i,
                &sk_in.data.as_vec_znx(),
                i,
            );
        });

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(self, sk_out.rank());
        {
            let (mut tmp, _) = scratch_2.take_scalar_znx(self, 1);
            (0..sk_out.rank().into()).for_each(|i| {
                self.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                self.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            });
        }

        res.key.encrypt_sk(
            self,
            &sk_in_tmp,
            &sk_out_tmp,
            source_xa,
            source_xe,
            scratch_2,
        );
        res.sk_in_n = sk_in.n().into();
        res.sk_out_n = sk_out.n().into();
    }    

}

pub trait GLWESwitchingKeyEncryptPk<BE: Backend> 
where
    Self: Sized
        + ModuleN
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + SvpPPolAlloc<BE>
        + GGLWEEncryptSk<BE>,
{

    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyEncryptPk<BE> for Module<BE> where
    Self: ModuleN
        + SvpPPolBytesOf
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + SvpPPolAlloc<BE>
        + GGLWEEncryptSk<BE>,
{

    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GGLWE::encrypt_pk_tmp_bytes(self, infos)
    }
}