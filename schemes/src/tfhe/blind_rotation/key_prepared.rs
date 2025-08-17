use backend::hal::{
    api::{SvpPPolAlloc, SvpPrepare, VmpPMatAlloc, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, ScalarZnx, Scratch, SvpPPol},
};

use std::marker::PhantomData;

use core::{
    Distribution,
    layouts::{
        Infos,
        prepared::{GGSWCiphertextPrepared, Prepare, PrepareAlloc},
    },
};

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKey, utils::set_xai_plus_y};

pub trait BlindRotationKeyPreparedAlloc<B: Backend> {
    fn alloc(module: &Module<B>, n_glwe: usize, n_lwe: usize, basek: usize, k: usize, rows: usize, rank: usize) -> Self;
}

#[derive(PartialEq, Eq)]
pub struct BlindRotationKeyPrepared<D: Data, BRT: BlindRotationAlgo, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertextPrepared<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPol<Vec<u8>, B>>>,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyPrepared<D, BRT, B> {
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

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>
    for BlindRotationKey<D, BRA>
where
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlindRotationKeyPreparedAlloc<B>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: Prepare<B, BlindRotationKey<D, BRA>>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> BlindRotationKeyPrepared<Vec<u8>, BRA, B> {
        let mut brk: BlindRotationKeyPrepared<Vec<u8>, BRA, B> = BlindRotationKeyPrepared::alloc(
            module,
            self.n(),
            self.keys.len(),
            self.basek(),
            self.k(),
            self.rows(),
            self.rank(),
        );
        brk.prepare(module, self, scratch);
        brk
    }
}

impl<DM: DataMut, DR: DataRef, BRA: BlindRotationAlgo, B: Backend> Prepare<B, BlindRotationKey<DR, BRA>>
    for BlindRotationKeyPrepared<DM, BRA, B>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B> + SvpPPolAlloc<B> + SvpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &BlindRotationKey<DR, BRA>, scratch: &mut Scratch<B>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.data.len(), other.keys.len());
        }

        let n: usize = other.n();

        self.data
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(ggsw_prepared, other)| {
                ggsw_prepared.prepare(module, other, scratch);
            });

        self.dist = other.dist;

        if let Distribution::BinaryBlock(_) = other.dist {
            let mut x_pow_a: Vec<SvpPPol<Vec<u8>, B>> = Vec::with_capacity(n << 1);
            let mut buf: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            (0..n << 1).for_each(|i| {
                let mut res: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(n, 1);
                set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                x_pow_a.push(res);
            });
            self.x_pow_a = Some(x_pow_a);
        }
    }
}
