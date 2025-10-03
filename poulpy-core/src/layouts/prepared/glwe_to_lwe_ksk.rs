use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Digits, GGLWEInfos, GLWEInfos, GLWEToLWEKey, LWEInfos, Rank, Rows, TorusPrecision,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEToLWESwitchingKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn digits(&self) -> Digits {
        self.0.digits()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn rows(&self) -> Rows {
        self.0.rows()
    }
}

impl<B: Backend> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        Self(GGLWESwitchingKeyPrepared::alloc(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_in: Rank) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GGLWESwitchingKeyPrepared::alloc_with(
            module,
            base2k,
            k,
            rows,
            Digits(1),
            rank_in,
            Rank(1),
        ))
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for GLWEToLWESwitchingKeyPrepared"
        );
        GGLWESwitchingKeyPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rows: Rows, rank_in: Rank) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, rows, Digits(1), rank_in, Rank(1))
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GLWEToLWESwitchingKeyPrepared<Vec<u8>, B>> for GLWEToLWEKey<D>
where
    Module<B>: VmpPrepare<B> + VmpPMatAlloc<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> {
        let mut ksk_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> = GLWEToLWESwitchingKeyPrepared::alloc(module, self);
        ksk_prepared.prepare(module, self, scratch);
        ksk_prepared
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, GLWEToLWEKey<DR>> for GLWEToLWESwitchingKeyPrepared<DM, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GLWEToLWEKey<DR>, scratch: &mut Scratch<B>) {
        self.0.prepare(module, &other.0, scratch);
    }
}
