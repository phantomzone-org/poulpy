use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataMut, DataRef, DeviceBuf, Module, Scratch},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEToRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
    LWEInfos, Rank, TorusPrecision,
    prepared::{GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut, GGLWEPreparedToRef},
};

/// DFT-domain (prepared) variant of a GLWE switching key.
///
/// Wraps a [`GGLWEPrepared`] with input/output degree metadata for
/// key-switching between GLWE ciphertexts. Tied to a specific backend
/// via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyPrepared<D: Data, B: Backend> {
    pub(crate) key: GGLWEPrepared<D, B>,
    pub(crate) input_degree: Degree,  // Degree of sk_in
    pub(crate) output_degree: Degree, // Degree of sk_out
}

impl<D: DataRef, BE: Backend> GLWESwitchingKeyDegrees for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&self) -> &Degree {
        &self.output_degree
    }

    fn input_degree(&self) -> &Degree {
        &self.input_degree
    }
}

impl<D: DataMut, BE: Backend> GLWESwitchingKeyDegreesMut for GLWESwitchingKeyPrepared<D, BE> {
    fn output_degree(&mut self) -> &mut Degree {
        &mut self.output_degree
    }

    fn input_degree(&mut self) -> &mut Degree {
        &mut self.input_degree
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn size(&self) -> usize {
        self.key.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWESwitchingKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }
}

pub trait GLWESwitchingKeyPreparedFactory<B: Backend>
where
    Self: GGLWEPreparedFactory<B>,
{
    fn glwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GLWESwitchingKeyPrepared<DeviceBuf<B>, B> {
        GLWESwitchingKeyPrepared::<DeviceBuf<B>, B> {
            key: self.gglwe_prepared_alloc(base2k, k, rank_in, rank_out, dnum, dsize),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESwitchingKeyPrepared<DeviceBuf<B>, B>
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepared_alloc(
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_glwe_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        self.gglwe_prepared_bytes_of(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    fn glwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.bytes_of_glwe_key_prepared(
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<B>)
    where
        R: GGLWEPreparedToMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        Scratch<B>: ScratchAvailable,
    {
        let res_infos = res.to_mut();
        assert!(
            scratch.available() >= self.glwe_switching_key_prepare_tmp_bytes(&res_infos),
            "scratch.available(): {} < GLWESwitchingKeyPreparedFactory::glwe_switching_key_prepare_tmp_bytes: {}",
            scratch.available(),
            self.glwe_switching_key_prepare_tmp_bytes(&res_infos)
        );
        self.gglwe_prepare(res, other, scratch);
        *res.input_degree() = *other.input_degree();
        *res.output_degree() = *other.output_degree();
    }
}

impl<B: Backend> GLWESwitchingKeyPreparedFactory<B> for Module<B> where Self: GGLWEPreparedFactory<B> {}

// module-only API: allocation/size helpers are provided by `GLWESwitchingKeyPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWESwitchingKeyPreparedFactory` on `Module`.

impl<D: DataRef, BE: Backend> GGLWEPreparedToRef<BE> for GLWESwitchingKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToRef<BE>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], BE> {
        self.key.to_ref()
    }
}

impl<D: DataRef, BE: Backend> GGLWEPreparedToMut<BE> for GLWESwitchingKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToMut<BE>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], BE> {
        self.key.to_mut()
    }
}
