use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWETensorKey, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc, PrepareScratchSpace},
};

#[derive(PartialEq, Eq)]
pub struct GGLWETensorKeyPrepared<D: Data, B: Backend> {
    pub(crate) keys: Vec<GGLWESwitchingKeyPrepared<D, B>>,
}

impl<D: Data, B: Backend> LWEInfos for GGLWETensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GGLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GGLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl<B: Backend> GGLWETensorKeyPrepared<Vec<u8>, B> {
    pub fn alloc<A>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAlloc<B>,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyPrepared"
        );
        Self::alloc_with(
            module,
            infos.base2k(),
            infos.k(),
            infos.dnum(),
            infos.dsize(),
            infos.rank_out(),
        )
    }

    pub fn alloc_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let mut keys: Vec<GGLWESwitchingKeyPrepared<Vec<u8>, B>> = Vec::new();
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKeyPrepared::alloc_with(
                module,
                base2k,
                k,
                Rank(1),
                rank,
                dnum,
                dsize,
            ));
        });
        Self { keys }
    }

    pub fn alloc_bytes<A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VmpPMatAllocBytes,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
        );
        let rank_out: usize = infos.rank_out().into();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs
            * GGLWESwitchingKeyPrepared::alloc_bytes_with(
                module,
                infos.base2k(),
                infos.k(),
                Rank(1),
                infos.rank_out(),
                infos.dnum(),
                infos.dsize(),
            )
    }

    pub fn alloc_bytes_with(module: &Module<B>, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * GGLWESwitchingKeyPrepared::alloc_bytes_with(module, base2k, k, Rank(1), rank, dnum, dsize)
    }
}

impl<D: DataMut, B: Backend> GGLWETensorKeyPrepared<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> GGLWETensorKeyPrepared<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GGLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<DR: DataRef, B: Backend, A: GGLWEInfos> PrepareScratchSpace<B, A> for GGLWETensorKeyPrepared<DR, B>
where
    GGLWESwitchingKeyPrepared<DR, B>: PrepareScratchSpace<B, A>,
{
    fn prepare_scratch_space(module: &Module<B>, infos: &A) -> usize {
        GGLWESwitchingKeyPrepared::<DR, B>::prepare_scratch_space(module, infos)
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Prepare<B, GGLWETensorKey<DR>> for GGLWETensorKeyPrepared<D, B>
where
    Module<B>: VmpPrepare<B>,
{
    fn prepare(&mut self, module: &Module<B>, other: &GGLWETensorKey<DR>, scratch: &mut Scratch<B>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), other.keys.len());
        }
        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(a, b)| {
                a.prepare(module, b, scratch);
            });
    }
}

impl<D: DataRef, B: Backend> PrepareAlloc<B, GGLWETensorKeyPrepared<Vec<u8>, B>> for GGLWETensorKey<D>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(&self, module: &Module<B>, scratch: &mut Scratch<B>) -> GGLWETensorKeyPrepared<Vec<u8>, B> {
        let mut tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = GGLWETensorKeyPrepared::alloc(module, self);
        tsk_prepared.prepare(module, self, scratch);
        tsk_prepared
    }
}
