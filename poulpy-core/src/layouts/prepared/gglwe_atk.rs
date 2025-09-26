use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Digits, GGLWEAutomorphismKey, GGLWELayoutInfos, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWEAutomorphismKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWESwitchingKeyPrepared<D, B>,
    pub(crate) p: i64,
}

impl<D: Data, B: Backend> GGLWEAutomorphismKeyPrepared<D, B> {
    pub fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, B: Backend> LWEInfos for GGLWEAutomorphismKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GGLWEAutomorphismKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWELayoutInfos for GGLWEAutomorphismKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn digits(&self) -> Digits {
        self.key.digits()
    }

    fn rows(&self) -> Rows {
        self.key.rows()
    }
}

impl<B: Backend> GGLWEAutomorphismKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWELayoutInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKeyPrepared"
        );
        GGLWEAutomorphismKeyPrepared::<Vec<u8>, B> {
            key: GGLWESwitchingKeyPrepared::alloc(module, infos),
            p: 0,
        }
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rows: Rows, digits: Digits, rank: Rank) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        GGLWEAutomorphismKeyPrepared {
            key: GGLWESwitchingKeyPrepared::alloc_with(module, base2k, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWELayoutInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKeyPrepared"
        );
        GGLWESwitchingKeyPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rows: Rows,
        digits: Digits,
        rank: Rank,
    ) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, rows, digits, rank, rank)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWEAutomorphismKey<DR>> for GGLWEAutomorphismKeyPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWEAutomorphismKey<DR>, scratch: &mut Scratch<B>) {
        self.key.prepare(module, &other.key, scratch);
        self.p = other.p;
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> for GGLWEAutomorphismKey<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWEAutomorphismKeyPrepared<Vec<u8>, B> {
        let mut atk_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> = GGLWEAutomorphismKeyPrepared::alloc(module, self);
        atk_prepared.prepare(module, self, scratch);
        atk_prepared
    }
}
