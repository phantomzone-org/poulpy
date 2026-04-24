use std::marker::PhantomData;

use poulpy_core::{
    Distribution, EncryptionInfos, GGSWCompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{GGSWCompressed, GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecret, LWESecretToRef},
};
use poulpy_hal::{
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToRef, ScratchArena, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::blind_rotation::{
    BlindRotationKeyCompressed, BlindRotationKeyCompressedEncryptSk, BlindRotationKeyCompressedFactory, BlindRotationKeyInfos,
    CGGI,
};

impl<D: HostDataRef> BlindRotationKeyCompressedFactory<CGGI> for BlindRotationKeyCompressed<D, CGGI> {
    fn blind_rotation_key_compressed_alloc<A>(infos: &A) -> BlindRotationKeyCompressed<Vec<u8>, CGGI>
    where
        A: BlindRotationKeyInfos,
    {
        let mut data: Vec<GGSWCompressed<Vec<u8>>> = Vec::with_capacity(infos.n_lwe().into());
        (0..infos.n_lwe().as_usize()).for_each(|_| data.push(GGSWCompressed::alloc_from_infos(infos)));
        BlindRotationKeyCompressed {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}

impl<BE: Backend> BlindRotationKeyCompressedEncryptSk<BE, CGGI> for Module<BE>
where
    Self: GGSWCompressedEncryptSk<BE>,
    // TODO(device): this implementation is still host-backed because the
    // compressed key path also stages plaintext limbs through host views.
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        self.ggsw_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn blind_rotation_key_compressed_encrypt_sk<'s, D, S0, S1, E>(
        &self,
        res: &mut BlindRotationKeyCompressed<D, CGGI>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        D: HostDataMut,
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef + LWEInfos + GetDistribution,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        assert_eq!(res.keys.len() as u32, sk_lwe.n());
        assert!(sk_glwe.n() <= self.n() as u32);
        assert_eq!(sk_glwe.rank(), res.rank());

        match sk_lwe.dist() {
            Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {}
            _ => {
                panic!("invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)")
            }
        }

        {
            let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();

            let mut source_xa: Source = Source::new(seed_xa);

            res.dist = sk_lwe.dist();

            let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n().into(), 1);
            let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data().to_ref();

            for (i, ggsw) in res.keys.iter_mut().enumerate() {
                pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
                let mut scratch_iter = scratch.borrow();
                self.ggsw_compressed_encrypt_sk(
                    ggsw,
                    &pt,
                    sk_glwe,
                    source_xa.new_seed(),
                    enc_infos,
                    source_xe,
                    &mut scratch_iter,
                );
            }
        }
    }
}
