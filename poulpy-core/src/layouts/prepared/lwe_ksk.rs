use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, LWESwitchingKey, Rank, TorusPrecision,
    prepared::GLWESwitchingKeyPrepared,
};

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

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
    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

impl<B: Backend> LWESwitchingKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
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
        Self(GLWESwitchingKeyPrepared::alloc(module, infos))
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GLWESwitchingKeyPrepared::alloc_with(
            module,
            base2k,
            k,
            Rank(1),
            Rank(1),
            dnum,
            Dsize(1),
        ))
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        debug_assert_eq!(
            infos.dsize().0,
            1,
            "dsize > 1 is not supported for LWESwitchingKey"
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
        GLWESwitchingKeyPrepared::alloc_bytes(module, infos)
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, Rank(1), Rank(1), dnum, Dsize(1))
    }
}

impl<B: Backend, A: GGLWEInfos> PrepareScratchSpace<B, A> for LWESwitchingKeyPrepared<Vec<u8>, B>
where
    GLWESwitchingKeyPrepared<Vec<u8>, B>: PrepareScratchSpace<B, A>,
{
    fn prepare_scratch_space(module: &Module<B>, infos: &A) -> usize {
        GLWESwitchingKeyPrepared::prepare_scratch_space(module, infos)
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
