use poulpy_hal::{
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution, GGSWEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GLWEInfos, GLWESecretPreparedToRef, LWEInfos, LWESecret, LWESecretToRef},
};

use crate::tfhe::blind_rotation::{
    BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyFactory, BlindRotationKeyInfos, CGGI,
};

impl<D: DataRef> BlindRotationKeyFactory<CGGI> for BlindRotationKey<D, CGGI> {
    fn blind_rotation_key_alloc<A>(infos: &A) -> BlindRotationKey<Vec<u8>, CGGI>
    where
        A: BlindRotationKeyInfos,
    {
        BlindRotationKey {
            keys: (0..infos.n_lwe().as_usize())
                .map(|_| GGSW::alloc_from_infos(infos))
                .collect(),
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}

impl<BE: Backend> BlindRotationKeyEncryptSk<BE, CGGI> for Module<BE>
where
    Self: GGSWEncryptSk<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A: GGSWInfos>(&self, infos: &A) -> usize {
        self.ggsw_encrypt_sk_tmp_bytes(infos)
    }

    fn blind_rotation_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut BlindRotationKey<D, CGGI>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        S0: GLWESecretPreparedToRef<BE> + GLWEInfos,
        S1: LWESecretToRef + LWEInfos + GetDistribution,
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

            res.dist = sk_lwe.dist();

            let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(sk_glwe.n().into(), 1);
            let sk_ref: ScalarZnx<&[u8]> = sk_lwe.data().to_ref();

            for (i, ggsw) in res.keys.iter_mut().enumerate() {
                pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
                ggsw.encrypt_sk(self, &pt, sk_glwe, source_xa, source_xe, scratch);
            }
        }
    }
}
