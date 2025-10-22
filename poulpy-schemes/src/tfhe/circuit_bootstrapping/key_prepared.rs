use poulpy_core::{
    layouts::{
        GGLWEInfos, GGSWInfos, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWETensorKeyLayout,
        GLWETensorKeyPreparedFactory, LWEInfos,
        prepared::{GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared},
    },
    trace_galois_elements,
};
use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::tfhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationKeyInfos, BlindRotationKeyLayout, BlindRotationKeyPrepared,
        BlindRotationKeyPreparedFactory,
    },
    circuit_bootstrapping::{CircuitBootstrappingKey, CircuitBootstrappingKeyInfos},
};

impl<BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
        M: CircuitBootstrappingKeyPreparedFactory<BRA, BE>,
    {
        module.circuit_bootstrapping_key_prepared_alloc_from_infos(infos)
    }
}

impl<D: DataMut, BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingKeyPrepared<D, BRA, BE> {
    pub fn prepare<DR, M>(&mut self, module: &M, other: &CircuitBootstrappingKey<DR, BRA>, scratch: &mut Scratch<BE>)
    where
        DR: DataRef,
        M: CircuitBootstrappingKeyPreparedFactory<BRA, BE>,
    {
        module.circuit_bootstrapping_key_prepare(self, other, scratch);
    }
}

impl<BE: Backend, BRA: BlindRotationAlgo> CircuitBootstrappingKeyPreparedFactory<BRA, BE> for Module<BE> where
    Self: Sized
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
{
}

pub trait CircuitBootstrappingKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend>
where
    Self: Sized
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
    {
        let atk_infos: &GLWEAutomorphismKeyLayout = &infos.atk_infos();
        let gal_els: Vec<i64> = trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64);

        CircuitBootstrappingKeyPrepared {
            brk: BlindRotationKeyPrepared::alloc(self, &infos.brk_infos()),
            tsk: GLWETensorKeyPrepared::alloc_from_infos(self, &infos.tsk_infos()),
            atk: gal_els
                .iter()
                .map(|&gal_el| {
                    let key = GLWEAutomorphismKeyPrepared::alloc_from_infos(self, atk_infos);
                    (gal_el, key)
                })
                .collect(),
        }
    }
    fn circuit_bootstrapping_key_prepare<DM, DR>(
        &self,
        res: &mut CircuitBootstrappingKeyPrepared<DM, BRA, BE>,
        other: &CircuitBootstrappingKey<DR, BRA>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR: DataRef,
    {
        res.brk.prepare(self, &other.brk, scratch);
        res.tsk.prepare(self, &other.tsk, scratch);

        for (k, a) in res.atk.iter_mut() {
            a.prepare(self, other.atk.get(k).unwrap(), scratch);
        }
    }
}

pub struct CircuitBootstrappingKeyPrepared<D: Data, BRA: BlindRotationAlgo, B: Backend> {
    pub(crate) brk: BlindRotationKeyPrepared<D, BRA, B>,
    pub(crate) tsk: GLWETensorKeyPrepared<Vec<u8>, B>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
}

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyPrepared<D, BRA, B> {
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

    fn tsk_infos(&self) -> GLWETensorKeyLayout {
        GLWETensorKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.k(),
            dnum: self.tsk.dnum(),
            dsize: self.tsk.dsize(),
            rank: self.tsk.rank(),
        }
    }
}
