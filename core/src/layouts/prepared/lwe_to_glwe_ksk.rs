use backend::hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes, VmpPMatPrepare},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat},
};

use crate::layouts::{Infos, LWEToGLWESwitchingKey, prepared::GGLWESwitchingKeyExec};

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKeyExec<D: Data, B: Backend>(pub(crate) GGLWESwitchingKeyExec<D, B>);

impl<D: Data, B: Backend> Infos for LWEToGLWESwitchingKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data, B: Backend> LWEToGLWESwitchingKeyExec<D, B> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

impl<B: Backend> LWEToGLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self
    where
        Module<B>: VmpPMatAlloc<B>,
    {
        Self(GGLWESwitchingKeyExec::alloc(
            module, n, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_out: usize) -> usize
    where
        Module<B>: VmpPMatAllocBytes,
    {
        GGLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, n, basek, k, rows, digits, 1, rank_out)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &LWEToGLWESwitchingKey<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        Module<B>: VmpPMatAlloc<B> + VmpPMatPrepare<B>,
    {
        let mut ksk_exec: LWEToGLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.n(),
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWEToGLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VmpPMatPrepare<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
