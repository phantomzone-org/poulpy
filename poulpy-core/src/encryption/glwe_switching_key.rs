use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPrepare, VecZnxSwitchRing},
    layouts::{Backend, DataMut, Module, ScalarZnx, Scratch},
    source::Source,
};

pub use crate::api::{GLWESwitchingKeyEncryptPk, GLWESwitchingKeyEncryptSk};
use crate::{
    EncryptionInfos, ScratchTakeCore,
    encryption::gglwe::GGLWEEncryptSk,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWEInfos, GLWESecret, GLWESecretToRef, GLWESwitchingKey, GLWESwitchingKeyDegreesMut, LWEInfos,
        prepared::GLWESecretPreparedFactory,
    },
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
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S1, S2, E, BE: Backend>(
        &mut self,
        module: &M,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        M: GLWESwitchingKeyEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_switching_key_encrypt_sk(self, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch);
    }
}

#[doc(hidden)]
pub trait GLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef;
}

impl<BE: Backend> GLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxSwitchRing + SvpPrepare<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = ScalarZnx::bytes_of(self.n(), infos.rank_in().into());
        let lvl_1: usize = self.bytes_of_glwe_secret_prepared_from_infos(infos);
        let lvl_2: usize = ScalarZnx::bytes_of(self.n(), 1).max(self.gglwe_encrypt_sk_tmp_bytes(infos));

        lvl_0 + lvl_1 + lvl_2
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        let sk_in: &GLWESecret<&[u8]> = &sk_in.to_ref();
        let sk_out: &GLWESecret<&[u8]> = &sk_out.to_ref();

        assert!(sk_in.n().0 <= self.n() as u32);
        assert!(sk_out.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_switching_key_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_switching_key_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_in_tmp, scratch_1) = scratch.take_scalar_znx(self.n(), sk_in.rank().into());
        for i in 0..sk_in.rank().into() {
            self.vec_znx_switch_ring(&mut sk_in_tmp.as_vec_znx_mut(), i, &sk_in.data.as_vec_znx(), i);
        }

        let (mut sk_out_tmp, scratch_2) = scratch_1.take_glwe_secret_prepared(self, sk_out.rank());
        {
            let (mut tmp, _) = scratch_2.take_scalar_znx(self.n(), 1);
            for i in 0..sk_out.rank().into() {
                self.vec_znx_switch_ring(&mut tmp.as_vec_znx_mut(), 0, &sk_out.data.as_vec_znx(), i);
                self.svp_prepare(&mut sk_out_tmp.data, i, &tmp, 0);
            }
        }

        sk_out_tmp.dist = sk_out.dist;

        self.gglwe_encrypt_sk(res, &sk_in_tmp, &sk_out_tmp, enc_infos, source_xe, source_xa, scratch_2);

        *res.input_degree() = sk_in.n();
        *res.output_degree() = sk_out.n();
    }
}

#[doc(hidden)]
pub trait GLWESwitchingKeyEncryptPkDefault<BE: Backend> {
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyEncryptPkDefault<BE> for Module<BE> {
    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(&self, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}
