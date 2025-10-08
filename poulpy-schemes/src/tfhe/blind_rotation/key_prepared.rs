use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPrepare, VmpPMatAlloc, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, ScalarZnx, Scratch, SvpPPol},
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
        prepared::{GGSWCiphertextPrepared, Prepare, PrepareAlloc},
    },
};

use crate::tfhe::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos, utils::set_xai_plus_y};

pub trait BlindRotationKeyPreparedAlloc<B: Backend> {
    fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: BlindRotationKeyInfos;
}

#[derive(PartialEq, Eq)]
pub struct BlindRotationKeyPrepared<D: Data, BRT: BlindRotationAlgo, B: Backend> {
    pub(crate) data: Vec<GGSWCiphertextPrepared<D, B>>,
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

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>
    for BlindRotationKey<D, BRA>
where
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlindRotationKeyPreparedAlloc<B>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: Prepare<B, BlindRotationKey<D, BRA>>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> BlindRotationKeyPrepared<Vec<u8>, BRA, B> {
        let mut brk: BlindRotationKeyPrepared<Vec<u8>, BRA, B> = BlindRotationKeyPrepared::alloc(module, self);
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

        let n: usize = other.n().as_usize();

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
                let mut res: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(1);
                set_xai_plus_y(module, i, 0, &mut res, &mut buf);
                x_pow_a.push(res);
            });
            self.x_pow_a = Some(x_pow_a);
        }
    }
}
