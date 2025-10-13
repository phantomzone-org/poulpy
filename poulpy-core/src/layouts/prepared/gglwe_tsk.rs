use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatAllocBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TensorKey, TensorKeyToRef, TorusPrecision,
    compressed::TensorKeyCompressedToMut,
    prepared::{
        GLWESwitchingKeyPrepare, GLWESwitchingKeyPrepareTmpBytes, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToMut,
        GLWESwitchingKeyPreparedToRef,
    },
};

#[derive(PartialEq, Eq)]
pub struct TensorKeyPrepared<D: Data, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKeyPrepared<D, B>>,
}

impl<D: Data, B: Backend> LWEInfos for TensorKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for TensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for TensorKeyPrepared<D, B> {
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

impl<B: Backend> TensorKeyPrepared<Vec<u8>, B> {
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
        let mut keys: Vec<GLWESwitchingKeyPrepared<Vec<u8>, B>> = Vec::new();
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyPrepared::alloc(
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
            * GLWESwitchingKeyPrepared::alloc_bytes(
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
        pairs * GLWESwitchingKeyPrepared::alloc_bytes(module, base2k, k, Rank(1), rank, dnum, dsize)
    }
}

impl<D: DataMut, B: Backend> TensorKeyPrepared<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef, B: Backend> TensorKeyPrepared<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKeyPrepared<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

pub trait TensorKeyPrepareTmpBytes {
    fn tensor_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize;
}

impl<B: Backend> TensorKeyPrepareTmpBytes for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepareTmpBytes,
{
    fn tensor_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize {
        self.glwe_switching_key_prepare_tmp_bytes(infos)
    }
}

impl<B: Backend> TensorKeyPrepared<Vec<u8>, B>
where
    Module<B>: TensorKeyPrepareTmpBytes,
{
    fn prepare_tmp_bytes<A>(&self, module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.tensor_key_prepare_tmp_bytes(infos)
    }
}

pub trait TensorKeyPrepare<B: Backend> {
    fn tensor_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: TensorKeyPreparedToMut<B>,
        O: TensorKeyToRef;
}

impl<B: Backend> TensorKeyPrepare<B> for Module<B>
where
    Module<B>: GLWESwitchingKeyPrepare<B>,
{
    fn tensor_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &Scratch<B>)
    where
        R: TensorKeyPreparedToMut<B>,
        O: TensorKeyToRef,
    {
        let res = res.to_mut();
        let other = other.to_ref();

        assert_eq!(self.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.glwe_switching_prepare(a, b, scratch);
        }
    }
}

impl<D: DataMut, B: Backend> TensorKeyPrepared<D, B>
where
    Module<B>: TensorKeyPrepare<B>,
{
    fn prepare<O>(&mut self, module: &Module<B>, other: &O, scratch: &mut Scratch<B>)
    where
        O: TensorKeyToRef,
    {
        module.tensor_key_prepare(self, other, scratch);
    }
}

pub trait TensorKeyPrepareAlloc<B: Backend> {
    fn tensor_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: TensorKeyToRef;
}

impl<B: Backend> TensorKeyPrepareAlloc<B> for Module<B>
where
    Module<B>: TensorKeyPrepare<B>,
{
    fn tensor_key_prepare_alloc<O>(&self, other: &O, scratch: &mut Scratch<B>)
    where
        O: TensorKeyToRef,
    {
        let mut ct_prepared: TensorKeyPrepared<Vec<u8>, B> = TensorKeyPrepared::alloc(self, other);
        self.tensor_key_prepare(ct_prepared, other, scratch);
        ct_prepared
    }
}

impl<D: DataRef> TensorKey<D> {
    pub fn prepare_alloc<B: Backend>(&self, module: &Module<B>, scratch: &Scratch<B>)
    where
        Module<B>: TensorKeyPrepareAlloc<B>,
    {
        module.tensor_key_prepare_alloc(self, scratch);
    }
}

pub trait TensorKeyPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> TensorKeyPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> TensorKeyPreparedToMut<B> for TensorKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToMut<B>,
{
    fn to_mut(&mut self) -> TensorKeyPrepared<&mut [u8], B> {
        TensorKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}

pub trait TensorKeyPreparedToRef<B: Backend> {
    fn to_ref(&self) -> TensorKeyPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> TensorKeyPreparedToRef<B> for TensorKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GLWESwitchingKeyPreparedToRef<B>,
{
    fn to_ref(&self) -> TensorKeyPrepared<&[u8], B> {
        TensorKeyPrepared {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}
