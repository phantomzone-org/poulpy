use backend::hal::layouts::{Backend, Data, SvpPPol};

use std::marker::PhantomData;

use core::{
    Distribution,
    layouts::{Infos, prepared::GGSWCiphertextExec},
};

use crate::tfhe::blind_rotation::BlindRotationAlgo;

#[derive(PartialEq, Eq)]
pub struct BlindRotationKeyExec<D: Data, BRT: BlindRotationAlgo, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertextExec<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPol<Vec<u8>, B>>>,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyExec<D, BRT, B> {
    #[allow(dead_code)]
    pub(crate) fn n(&self) -> usize {
        self.data[0].n()
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.data[0].rows()
    }

    #[allow(dead_code)]
    pub(crate) fn k(&self) -> usize {
        self.data[0].k()
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        self.data[0].size()
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.data[0].rank()
    }

    pub(crate) fn basek(&self) -> usize {
        self.data[0].basek()
    }

    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}
