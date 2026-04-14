use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataMut, DataRef, DeviceBuf, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut, GGLWEPreparedToRef,
    GGLWEToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of a GLWE tensor key.
///
/// A newtype wrapper around [`GGLWEPrepared`] for tensor operations.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWETensorKeyPrepared<D: Data, B: Backend>(pub(crate) GGLWEPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWETensorKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.0.n()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWETensorKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

pub trait GLWETensorKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn alloc_tensor_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> GLWETensorKeyPrepared<DeviceBuf<B>, B> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyPrepared(self.alloc_gglwe_prepared(base2k, k, Rank(pairs), rank, dnum, dsize))
    }

    fn alloc_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> GLWETensorKeyPrepared<DeviceBuf<B>, B>
    where
        A: GGLWEInfos,
    {
        self.alloc_tensor_key_prepared(infos.base2k(), infos.max_k(), infos.dnum(), infos.dsize(), infos.rank_out())
    }

    fn bytes_of_tensor_key_prepared(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        self.bytes_of_gglwe_prepared(base2k, k, Rank(pairs), rank, dnum, dsize)
    }

    fn bytes_of_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_tensor_key_prepared(infos.base2k(), infos.max_k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn prepare_tensor_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.prepare_gglwe_tmp_bytes(infos);
        lvl_0
    }

    fn prepare_tensor_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B>,
        O: GGLWEToRef,
        Scratch<B>: ScratchAvailable,
    {
        let res_infos = res.to_mut();
        assert!(
            scratch.available() >= self.prepare_tensor_key_tmp_bytes(&res_infos),
            "scratch.available(): {} < GLWETensorKeyPreparedFactory::prepare_tensor_key_tmp_bytes: {}",
            scratch.available(),
            self.prepare_tensor_key_tmp_bytes(&res_infos)
        );
        self.prepare_gglwe(res, other, scratch);
    }
}

impl<B: Backend> GLWETensorKeyPreparedFactory<B> for Module<B> where Module<B>: GGLWEPreparedFactory<B> {}

// module-only API: allocation/size helpers are provided by `GLWETensorKeyPreparedFactory` on `Module`.

// module-only API: preparation sizing is provided by `GLWETensorKeyPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWETensorKeyPreparedFactory` on `Module`.

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for GLWETensorKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}
