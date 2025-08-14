use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{GGLWETensorKey, Infos, prepared::GGLWESwitchingKeyExec};

#[derive(PartialEq, Eq)]
pub struct GGLWETensorKeyExec<D: Data, B: Backend> {
    pub(crate) keys: Vec<GGLWESwitchingKeyExec<D, B>>,
}

impl<B: Backend> GGLWETensorKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        let mut keys: Vec<GGLWESwitchingKeyExec<Vec<u8>, B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKeyExec::alloc(
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
        pairs * GGLWESwitchingKeyExec::bytes_of(module, n, basek, k, rows, digits, 1, rank)
    }

    pub fn from<D: DataRef>(
        module: &Module<B>,
        other: &GGLWETensorKey<D>,
        scratch: &mut Scratch<B>,
    ) -> GGLWETensorKeyExec<Vec<u8>, B>
    where
        Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut tsk_exec: GGLWETensorKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.n(),
            other.basek(),
            other.k(),
            other.rows(),
            other.digits(),
            other.rank(),
        );
        tsk_exec.prepare(module, other, scratch);
        tsk_exec
    }
}

impl<D: Data, B: Backend> Infos for GGLWETensorKeyExec<D, B> {
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

impl<D: Data, B: Backend> GGLWETensorKeyExec<D, B> {
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

impl<D: DataMut, B: Backend> GGLWETensorKeyExec<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> GGLWETensorKeyExec<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GGLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut, B: Backend> GGLWETensorKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GGLWETensorKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
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
