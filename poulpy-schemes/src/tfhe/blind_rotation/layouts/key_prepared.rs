use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Scratch, SvpPPol};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Dnum, Dsize, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision, prepared::GGSWPrepared},
};

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos};

pub trait BlindRotationKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    fn blind_rotation_key_prepared_alloc<A>(&self, infos: &A) -> BlindRotationKeyPrepared<Vec<u8>, BRA, BE>
    where
        A: BlindRotationKeyInfos;

    fn blind_rotation_key_prepare<DM, DR>(
        &self,
        res: &mut BlindRotationKeyPrepared<DM, BRA, BE>,
        other: &BlindRotationKey<DR, BRA>,
        scratch: &mut Scratch<BE>,
    ) where
        DM: DataMut,
        DR: DataRef;
}

impl<BE: Backend, BRA: BlindRotationAlgo> BlindRotationKeyPrepared<Vec<u8>, BRA, BE> {
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        A: BlindRotationKeyInfos,
        M: BlindRotationKeyPreparedFactory<BRA, BE>,
    {
        module.blind_rotation_key_prepared_alloc(infos)
    }
}

impl<D: DataMut, BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<D, BRA, BE> {
    pub fn prepare<DR: DataRef, M>(&mut self, module: &M, other: &BlindRotationKey<DR, BRA>, scratch: &mut Scratch<BE>)
    where
        M: BlindRotationKeyPreparedFactory<BRA, BE>,
    {
        module.blind_rotation_key_prepare(self, other, scratch);
    }
}

#[derive(PartialEq, Eq)]
pub struct BlindRotationKeyPrepared<D: Data, BRT: BlindRotationAlgo, B: Backend> {
    pub(crate) data: Vec<GGSWPrepared<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPol<Vec<u8>, B>>>,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.data.len() as u32)
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> LWEInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn base2k(&self) -> Base2K {
        self.data[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.data[0].k()
    }

    fn n(&self) -> Degree {
        self.data[0].n()
    }

    fn size(&self) -> usize {
        self.data[0].size()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GLWEInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn rank(&self) -> Rank {
        self.data[0].rank()
    }
}
impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GGSWInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.data[0].dnum()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyPrepared<D, BRT, B> {
    pub fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}
