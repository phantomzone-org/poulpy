use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{Backend, Data, DataMut, DataRef, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos},
};

use crate::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToRef, GetDegree, LWEInfos, Rank, TorusPrecision};

#[derive(PartialEq, Eq)]
pub struct GLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
}

impl<D: Data, B: Backend> LWEInfos for GLWEPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

pub trait GLWEPreparedAlloc<B: Backend>
where
    Self: GetDegree + VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B>,
{
    fn alloc_glwe_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPrepared<Vec<u8>, B> {
        GLWEPrepared {
            data: self.vec_znx_dft_alloc((rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
        }
    }

    fn alloc_glwe_prepared_from_infos<A>(&self, infos: &A) -> GLWEPrepared<Vec<u8>, B>
    where
        A: GLWEInfos,
    {
        self.alloc_glwe_prepared(infos.base2k(), infos.k(), infos.rank())
    }

    fn bytes_of_glwe_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.bytes_of_vec_znx_dft((rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    fn bytes_of_glwe_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_glwe_prepared(infos.base2k(), infos.k(), infos.rank())
    }

    fn prepare_glwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPreparedToMut<B>,
        O: GLWEToRef,
    {
        {
            let mut res: GLWEPrepared<&mut [u8], B> = res.to_mut();
            let other: GLWE<&[u8]> = other.to_ref();

            assert_eq!(res.n(), self.ring_degree());
            assert_eq!(other.n(), self.ring_degree());
            assert_eq!(res.size(), other.size());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.base2k(), other.base2k());

            for i in 0..(res.rank() + 1).into() {
                self.vec_znx_dft_apply(1, 0, &mut res.data, i, &other.data, i);
            }
        }
    }
}

impl<B: Backend> GLWEPreparedAlloc<B> for Module<B> where Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B> {}

impl<B: Backend> GLWEPrepared<Vec<u8>, B> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GLWEInfos,
        M: GLWEPreparedAlloc<B>,
    {
        module.alloc_glwe_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: GLWEPreparedAlloc<B>,
    {
        module.alloc_glwe_prepared(base2k, k, rank)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWEPreparedAlloc<B>,
    {
        module.bytes_of_glwe_prepared_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize
    where
        M: GLWEPreparedAlloc<B>,
    {
        module.bytes_of_glwe_prepared(base2k, k, rank)
    }
}

impl<D: DataMut, B: Backend> GLWEPrepared<D, B> {
    pub fn prepare<O, M>(&mut self, module: &M, other: &O)
    where
        O: GLWEToRef,
        M: GLWEPreparedAlloc<B>,
    {
        module.prepare_glwe(self, other);
    }
}

pub trait GLWEPreparedToMut<B: Backend> {
    fn to_mut(&mut self) -> GLWEPrepared<&mut [u8], B>;
}

impl<D: DataMut, B: Backend> GLWEPreparedToMut<B> for GLWEPrepared<D, B> {
    fn to_mut(&mut self) -> GLWEPrepared<&mut [u8], B> {
        GLWEPrepared {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

pub trait GLWEPreparedToRef<B: Backend> {
    fn to_ref(&self) -> GLWEPrepared<&[u8], B>;
}

impl<D: DataRef, B: Backend> GLWEPreparedToRef<B> for GLWEPrepared<D, B> {
    fn to_ref(&self) -> GLWEPrepared<&[u8], B> {
        GLWEPrepared {
            data: self.data.to_ref(),
            k: self.k,
            base2k: self.base2k,
        }
    }
}
