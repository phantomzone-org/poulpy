use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPrepare, VecZnxSwitchRing},
    layouts::{Backend, DataMut, Module, ScalarZnx, Scratch},
    source::Source,
};

use crate::{
    GGLWECompressedEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretToRef,
        GLWESwitchingKeyDegreesMut, LWEInfos,
        compressed::GLWESwitchingKeyCompressed,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GLWESwitchingKeyCompressedEncryptSk<BE>,
    {
        module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GLWESwitchingKeyCompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S1, S2, BE: Backend>(
        &mut self,
        module: &M,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
        M: GLWESwitchingKeyCompressedEncryptSk<BE>,
    {
        module.glwe_switching_key_compressed_encrypt_sk(self, sk_in, sk_out, seed_xa, source_xe, scratch);
    }
}

pub trait GLWESwitchingKeyCompressedEncryptSk<BE: Backend> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef;
}

impl<BE: Backend> GLWESwitchingKeyCompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxSwitchRing,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = ScalarZnx::bytes_of(self.n(), infos.rank_in().into());
        let lvl_1: usize = GLWESecretPrepared::bytes_of(self, infos.rank_out());
        let lvl_2: usize = ScalarZnx::bytes_of(self.n(), 1).max(self.gglwe_compressed_encrypt_sk_tmp_bytes(infos));

        lvl_0 + lvl_1 + lvl_2
    }

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef,
    {
        let sk_in: &GLWESecret<&[u8]> = &sk_in.to_ref();
        let sk_out: &GLWESecret<&[u8]> = &sk_out.to_ref();

        assert!(sk_in.n().0 <= self.n() as u32);
        assert!(sk_out.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWESwitchingKeyCompressedEncryptSk::glwe_switching_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(res)
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

        self.gglwe_compressed_encrypt_sk(res, &sk_in_tmp, &sk_out_tmp, seed_xa, source_xe, scratch_2);

        *res.input_degree() = sk_in.n();
        *res.output_degree() = sk_out.n();
    }
}
