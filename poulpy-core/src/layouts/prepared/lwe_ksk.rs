use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Digits, GGLWEInfos, GLWEInfos, LWEInfos, LWESwitchingKey, Rank, Rows, TorusPrecision,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWESwitchingKeyPrepared<D, B> {
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
impl<D: Data, B: Backend> GLWEInfos for LWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for LWESwitchingKeyPrepared<D, B> {
    fn digits(&self) -> Digits {
        self.0.digits()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn rows(&self) -> Rows {
        self.0.rows()
    }
}

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        Self(GGLWESwitchingKeyPrepared::alloc(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rows: Rows) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GGLWESwitchingKeyPrepared::alloc_with(
            module,
            base2k,
            k,
            rows,
            Digits(1),
            Rank(1),
            Rank(1),
        ))
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(
            infos.digits().0,
            1,
            "digits > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_in().0,
            1,
            "rank_in > 1 is not supported for LWESwitchingKey"
        );
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for LWESwitchingKey"
        );
        GGLWESwitchingKeyPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rows: Rows) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, rows, Digits(1), Rank(1), Rank(1))
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, LWESwitchingKeyPrepared<Vec<u8>, B>> for LWESwitchingKey<D>
where
    Module<B>: VmpPrepare<B> + VmpPMatAlloc<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> LWESwitchingKeyPrepared<Vec<u8>, B> {
        let mut ksk_prepared: LWESwitchingKeyPrepared<Vec<u8>, B> = LWESwitchingKeyPrepared::alloc(module, self);
        ksk_prepared.prepare(module, self, scratch);
        ksk_prepared
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, LWESwitchingKey<DR>> for LWESwitchingKeyPrepared<DM, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &LWESwitchingKey<DR>, scratch: &mut Scratch<B>) {
        self.0.prepare(module, &other.0, scratch);
    }
}
