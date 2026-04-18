use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataMut, DataRef, DeviceBuf, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedToMut, GGLWEPreparedToRef, GGLWEToRef, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory},
};

/// DFT-domain (prepared) variant of an LWE switching key.
///
/// A newtype wrapper around [`GLWESwitchingKeyPrepared`] for LWE key-switching.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for LWESwitchingKeyPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
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

pub trait LWESwitchingKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn lwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
    ) -> LWESwitchingKeyPrepared<DeviceBuf<B>, B> {
        LWESwitchingKeyPrepared(self.glwe_switching_key_prepared_alloc(base2k, k, Rank(1), Rank(1), dnum, Dsize(1)))
    }

    fn lwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWESwitchingKeyPrepared<DeviceBuf<B>, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        self.lwe_switching_key_prepared_alloc(infos.base2k(), infos.max_k(), infos.dnum())
    }

    fn lwe_switching_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, k, Rank(1), Rank(1), dnum, Dsize(1))
    }

    fn lwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        self.lwe_switching_key_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.dnum())
    }

    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_switching_key_prepare_tmp_bytes(infos);
        lvl_0
    }
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        Scratch<B>: ScratchAvailable,
    {
        let res_infos = res.to_mut();
        assert!(
            scratch.available() >= self.lwe_switching_key_prepare_tmp_bytes(&res_infos),
            "scratch.available(): {} < LWESwitchingKeyPreparedFactory::lwe_switching_key_prepare_tmp_bytes: {}",
            scratch.available(),
            self.lwe_switching_key_prepare_tmp_bytes(&res_infos)
        );
        self.glwe_switching_key_prepare(res, other, scratch);
    }
}

impl<B: Backend> LWESwitchingKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `LWESwitchingKeyPreparedFactory` on `Module`.

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for LWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for LWESwitchingKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToMut<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyDegreesMut for LWESwitchingKeyPrepared<D, B> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}
