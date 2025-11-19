use poulpy_core::{
    Distribution, GGLWEToGGSWKeyEncryptSk, GLWEAutomorphismKeyEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGSWInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEInfos,
        GLWESecretPreparedFactory, GLWESecretToRef, LWEInfos, LWESecretToRef, prepared::GLWESecretPrepared,
    },
    trace_galois_elements,
};
use std::collections::HashMap;

use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::bin_fhe::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyFactory, BlindRotationKeyInfos,
    BlindRotationKeyLayout,
};

pub trait CircuitBootstrappingKeyInfos {
    fn block_size(&self) -> usize;
    fn brk_infos(&self) -> BlindRotationKeyLayout;
    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout;
    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout;
}

#[derive(Debug, Clone, Copy)]
pub struct CircuitBootstrappingKeyLayout {
    pub layout_brk: BlindRotationKeyLayout,
    pub layout_atk: GLWEAutomorphismKeyLayout,
    pub layout_tsk: GGLWEToGGSWKeyLayout,
}

impl CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyLayout {
    fn block_size(&self) -> usize {
        unimplemented!("unimplemented for CircuitBootstrappingKeyLayout")
    }

    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout {
        self.layout_atk
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        self.layout_brk
    }

    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout {
        self.layout_tsk
    }
}

pub trait CircuitBootstrappingKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend> {
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;

    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<D, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GLWEInfos + GetDistribution;
}

impl<BRA: BlindRotationAlgo> CircuitBootstrappingKey<Vec<u8>, BRA> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: CircuitBootstrappingKeyInfos,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    {
        let atk_infos: &GLWEAutomorphismKeyLayout = &infos.atk_infos();
        let brk_infos: &BlindRotationKeyLayout = &infos.brk_infos();
        let trk_infos: &GGLWEToGGSWKeyLayout = &infos.tsk_infos();
        let gal_els: Vec<i64> = trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64);

        Self {
            brk: <BlindRotationKey<Vec<u8>, BRA> as BlindRotationKeyFactory<BRA>>::blind_rotation_key_alloc(brk_infos),
            atk: gal_els
                .iter()
                .map(|&gal_el| {
                    let key = GLWEAutomorphismKey::alloc_from_infos(atk_infos);
                    (gal_el, key)
                })
                .collect(),
            tsk: GGLWEToGGSWKey::alloc_from_infos(trk_infos),
        }
    }
}

pub struct CircuitBootstrappingKey<D: Data, BRA: BlindRotationAlgo> {
    pub(crate) brk: BlindRotationKey<D, BRA>,
    pub(crate) tsk: GGLWEToGGSWKey<Vec<u8>>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKey<Vec<u8>>>,
}

impl<D: DataMut, BRA: BlindRotationAlgo> CircuitBootstrappingKey<D, BRA> {
    pub fn encrypt_sk<M, S0, S1, BE: Backend>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GLWEInfos + GetDistribution,
        M: CircuitBootstrappingKeyEncryptSk<BRA, BE>,
    {
        module.circuit_bootstrapping_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xa, source_xe, scratch);
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingKeyEncryptSk<BRA, BE> for Module<BE>
where
    Self: GGLWEToGGSWKeyEncryptSk<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
    {
        self.glwe_automorphism_key_encrypt_sk_tmp_bytes(&infos.atk_infos())
            .max(self.blind_rotation_key_encrypt_sk_tmp_bytes(&infos.brk_infos()))
            .max(self.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&infos.tsk_infos()))
    }

    fn circuit_bootstrapping_key_encrypt_sk<D, S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<D, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        D: DataMut,
        S0: LWESecretToRef + GetDistribution + LWEInfos,
        S1: GLWESecretToRef + GLWEInfos + GetDistribution,
    {
        let brk_infos: &BlindRotationKeyLayout = &res.brk_infos();
        let atk_infos: &GLWEAutomorphismKeyLayout = &res.atk_infos();
        let tsk_infos: &GGLWEToGGSWKeyLayout = &res.tsk_infos();

        assert_eq!(sk_lwe.n(), brk_infos.n_lwe());
        assert_eq!(sk_glwe.n(), brk_infos.n_glwe());
        assert_eq!(sk_glwe.n(), atk_infos.n());
        assert_eq!(sk_glwe.n(), tsk_infos.n());

        assert!(sk_glwe.dist() != &Distribution::NONE);

        for (p, atk) in res.atk.iter_mut() {
            atk.encrypt_sk(self, *p, sk_glwe, source_xa, source_xe, scratch);
        }

        let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(self, brk_infos.rank());
        sk_glwe_prepared.prepare(self, sk_glwe);

        res.brk.encrypt_sk(
            self,
            &sk_glwe_prepared,
            sk_lwe,
            source_xa,
            source_xe,
            scratch,
        );

        res.tsk
            .encrypt_sk(self, sk_glwe, source_xa, source_xe, scratch);
    }
}

impl<D: DataRef, BRA: BlindRotationAlgo> CircuitBootstrappingKeyInfos for CircuitBootstrappingKey<D, BRA> {
    fn block_size(&self) -> usize {
        self.brk.block_size()
    }

    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout {
        let (_, atk) = self.atk.iter().next().expect("atk is empty");
        GLWEAutomorphismKeyLayout {
            n: atk.n(),
            base2k: atk.base2k(),
            k: atk.k(),
            dnum: atk.dnum(),
            dsize: atk.dsize(),
            rank: atk.rank(),
        }
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        BlindRotationKeyLayout {
            n_glwe: self.brk.n_glwe(),
            n_lwe: self.brk.n_lwe(),
            base2k: self.brk.base2k(),
            k: self.brk.k(),
            dnum: self.brk.dnum(),
            rank: self.brk.rank(),
        }
    }

    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout {
        GGLWEToGGSWKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.k(),
            dnum: self.tsk.dnum(),
            dsize: self.tsk.dsize(),
            rank: self.tsk.rank(),
        }
    }
}
