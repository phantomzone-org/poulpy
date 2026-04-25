use poulpy_hal::{
    api::{ScratchOwnedAlloc, SvpPrepare},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned, SvpPPolToBackendMut},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWELayout, GLWEInfos, GLWESecretPreparedFactory,
        GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToRef,
    },
};

#[doc(hidden)]
pub trait GLWETensorKeyCompressedEncryptSkDefault<BE: Backend> {
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: HostDataMut,
{
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = self.glwe_secret_prepared_bytes_of(infos.rank_out());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);

        let tensor_infos: GGLWELayout = GGLWELayout {
            n: infos.n(),
            base2k: infos.base2k(),
            k: infos.max_k(),
            rank_in: GLWESecretTensor::pairs(infos.rank().into()).into(),
            rank_out: infos.rank_out(),
            dnum: infos.dnum(),
            dsize: infos.dsize(),
        };

        let lvl_0: usize = sk_prepared;
        let lvl_1: usize = sk_tensor;
        let lvl_2_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(&tensor_infos);
        let lvl_2_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_2: usize = lvl_2_encrypt.max(lvl_2_prepare);

        lvl_0 + lvl_1 + lvl_2
    }

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEInfos + GGLWECompressedToMut + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert!(
            scratch.available()
                >= <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(
                    self, res
                ),
            "scratch.available(): {} < GLWETensorKeyCompressedEncryptSk::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            <Module<BE> as GLWETensorKeyCompressedEncryptSkDefault<BE>>::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(
                self, res
            )
        );

        let mut sk_prepared = self.glwe_secret_prepared_alloc(res.rank());
        let mut sk_tensor = GLWESecretTensor::alloc(self.n().into(), res.rank());
        {
            let sk_ref = sk.to_ref();
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk_ref.data.data), sk_ref.data.n, sk_ref.data.cols);
            let sk_backend_ref = <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&sk_backend);
            let mut sk_prepared_data = sk_prepared.data.to_backend_mut();
            for i in 0..sk_ref.rank().into() {
                self.svp_prepare(&mut sk_prepared_data, i, &sk_backend_ref, i);
            }
            sk_prepared.dist = *sk.dist();
        }
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, scratch);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));
        self.gglwe_compressed_encrypt_sk(
            res,
            &ScalarZnx::from_data(
                BE::from_host_bytes(sk_tensor.data.to_ref().data),
                sk_tensor.data.n,
                sk_tensor.data.cols,
            ),
            &sk_prepared,
            seed_xa,
            enc_infos,
            source_xe,
            &mut enc_scratch.arena(),
        );
    }
}
