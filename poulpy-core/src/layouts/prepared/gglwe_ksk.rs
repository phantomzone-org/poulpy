use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWESwitchingKey, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWECiphertextPrepared, Prepare, PrepareAlloc, PrepareScratchSpace},
};

#[derive(PartialEq, Eq)]
pub struct GGLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWECiphertextPrepared<D, B>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: Data, B: Backend> LWEInfos for GGLWESwitchingKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GGLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GGLWESwitchingKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

impl<B: Backend> GGLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(module.n() as u32, infos.n(), "module.n() != infos.n()");
        GGLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: GGLWECiphertextPrepared::alloc(module, infos),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWESwitchingKeyPrepared::<Vec<u8>, B> {
            key: GGLWECiphertextPrepared::alloc_with(module, base2k, k, rank_in, rank_out, dnum, dsize),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(module.n() as u32, infos.n(), "module.n() != infos.n()");
        GGLWECiphertextPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWECiphertextPrepared::alloc_bytes_with(module, base2k, k, rank_in, rank_out, dnum, dsize)
    }
}

impl<DR: DataRef, B: Backend, A: GGLWEInfos> PrepareScratchSpace<B, A> for GGLWESwitchingKeyPrepared<DR, B>
where
    GGLWECiphertextPrepared<DR, B>: PrepareScratchSpace<B, A>,
{
    fn prepare_scratch_space(module: &Module<B>, infos: &A) -> usize {
        GGLWECiphertextPrepared::<DR, B>::prepare_scratch_space(module, infos)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWESwitchingKey<DR>> for GGLWESwitchingKeyPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWESwitchingKey<DR>, scratch: &mut Scratch<B>) {
        self.key.prepare(module, &other.key, scratch);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWESwitchingKeyPrepared<Vec<u8>, B>> for GGLWESwitchingKey<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWESwitchingKeyPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWESwitchingKeyPrepared<Vec<u8>, B> = GGLWESwitchingKeyPrepared::alloc(module, self);
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
