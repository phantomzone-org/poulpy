use backend::hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VmpPMat};

use crate::{
    GGLWEExecLayoutFamily, GLWESwitchingKeyExec, Infos,
    lwe::keyswtich_layouts::{GLWEToLWESwitchingKey, LWESwitchingKey, LWEToGLWESwitchingKey},
};

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyExec<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyExec<D, B>);

impl<D: Data, B: Backend> Infos for GLWEToLWESwitchingKeyExec<D, B> {
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

impl<D: Data, B: Backend> GLWEToLWESwitchingKeyExec<D, B> {
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

impl<B: Backend> GLWEToLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, n, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, n, basek, k, rows, digits, rank_in, 1)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &GLWEToLWESwitchingKey<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: GLWEToLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.n(),
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank_in(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> GLWEToLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWEToLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKeyExec<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyExec<D, B>);

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
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, n, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, n, basek, k, rows, digits, 1, rank_out)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &LWEToGLWESwitchingKey<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
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
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyExec<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyExec<D, B>);

impl<D: Data, B: Backend> Infos for LWESwitchingKeyExec<D, B> {
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

impl<D: Data, B: Backend> LWESwitchingKeyExec<D, B> {
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

impl<B: Backend> LWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, n, basek, k, rows, 1, 1, 1,
        ))
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rows: usize, digits: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, n, basek, k, rows, digits, 1, 1)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: LWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.n(),
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> LWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
