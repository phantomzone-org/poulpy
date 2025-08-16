use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{
    GGLWETensorKey, Infos,
    prepared::{GGLWESwitchingKeyPrepared, Prepare, PrepareAlloc},
};

#[derive(PartialEq, Eq)]
pub struct GGLWETensorKeyPrepared<D: Data, B: Backend> {
    pub(crate) keys: Vec<GGLWESwitchingKeyPrepared<D, B>>,
}

impl<B: Backend> GGLWETensorKeyPrepared<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let mut keys: Vec<GGLWESwitchingKeyPrepared<Vec<u8>, B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKeyPrepared::alloc(
                module, n, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GGLWESwitchingKeyPrepared::bytes_of(module, n, basek, k, rows, digits, 1, rank)
    }
}

impl<D: Data, B: Backend> Infos for GGLWETensorKeyPrepared<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.keys[0].inner()
    }

    fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    fn k(&self) -> usize {
        self.keys[0].k()
    }
}

impl<D: Data, B: Backend> GGLWETensorKeyPrepared<D, B> {
    pub fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub fn rank_in(&self) -> usize {
        self.keys[0].rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.keys[0].rank_out()
    }

    pub fn digits(&self) -> usize {
        self.keys[0].digits()
    }
}

impl<D: DataMut, B: Backend> GGLWETensorKeyPrepared<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> GGLWETensorKeyPrepared<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GGLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
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
        let mut tsk_prepared: GGLWETensorKeyPrepared<Vec<u8>, B> = GGLWETensorKeyPrepared::alloc(
            module,
            self.n(),
            self.basek(),
            self.k(),
            self.rows(),
            self.digits(),
            self.rank(),
        );
        tsk_prepared.prepare(module, self, scratch);
        tsk_prepared
    }
}
