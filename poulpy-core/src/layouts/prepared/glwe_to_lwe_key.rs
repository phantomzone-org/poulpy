use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, DataMut, DataRef, Module, ScratchArena},
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GGLWEPreparedToMut, GGLWEPreparedToRef, GGLWEToRef, GLWEInfos, GLWESwitchingKeyDegrees,
    GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{
        GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWESwitchingKeyPreparedToBackendMut,
        GLWESwitchingKeyPreparedToBackendRef,
    },
};

/// DFT-domain (prepared) variant of a GLWE-to-LWE conversion key.
///
/// A newtype wrapper around [`GLWESwitchingKeyPrepared`] for converting
/// GLWE to LWE. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWEToLWEKeyPrepared<D: Data, B: Backend>(pub(crate) GLWESwitchingKeyPrepared<D, B>);

impl<D: Data, B: Backend> LWEInfos for GLWEToLWEKeyPrepared<D, B> {
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

impl<D: Data, B: Backend> GLWEInfos for GLWEToLWEKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend> GGLWEInfos for GLWEToLWEKeyPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

pub trait GLWEToLWEKeyPreparedFactory<B: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<B>,
{
    fn glwe_to_lwe_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
    ) -> GLWEToLWEKeyPrepared<B::OwnedBuf, B> {
        GLWEToLWEKeyPrepared(self.glwe_switching_key_prepared_alloc(base2k, k, rank_in, Rank(1), dnum, Dsize(1)))
    }
    fn glwe_to_lwe_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEToLWEKeyPrepared<B::OwnedBuf, B>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWEKeyPrepared"
        );
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for GLWEToLWEKeyPrepared");
        self.glwe_to_lwe_key_prepared_alloc(infos.base2k(), infos.max_k(), infos.rank_in(), infos.dnum())
    }

    fn glwe_to_lwe_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, k, rank_in, Rank(1), dnum, Dsize(1))
    }

    fn glwe_to_lwe_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWEKeyPrepared"
        );
        debug_assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for GLWEToLWEKeyPrepared");
        self.glwe_to_lwe_key_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.rank_in(), infos.dnum())
    }

    fn glwe_to_lwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_switching_key_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn glwe_to_lwe_key_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, B>)
    where
        R: GGLWEPreparedToMut<B> + GGLWEPreparedToBackendMut<B> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToRef + GLWESwitchingKeyDegrees,
        ScratchArena<'s, B>: ScratchAvailable,
        B: 's,
    {
        let tmp_bytes = {
            let res_infos = res.to_mut();
            self.glwe_to_lwe_key_prepare_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GLWEToLWEKeyPreparedFactory::glwe_to_lwe_key_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        self.glwe_switching_key_prepare(res, other, scratch);
    }
}

impl<B: Backend> GLWEToLWEKeyPreparedFactory<B> for Module<B> where Self: GLWESwitchingKeyPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEToLWEKeyPreparedFactory` on `Module`.

impl<D: DataRef, B: Backend> GGLWEPreparedToRef<B> for GLWEToLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_ref(&self) -> GGLWEPrepared<&[u8], B> {
        self.0.to_ref()
    }
}

impl<B: Backend> GGLWEPreparedToBackendMut<B> for GLWEToLWEKeyPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> crate::layouts::GGLWEPreparedBackendMut<'_, B> {
        GLWESwitchingKeyPreparedToBackendMut::to_backend_mut(&mut self.0).key
    }
}

impl<D: DataMut, B: Backend> GGLWEPreparedToMut<B> for GLWEToLWEKeyPrepared<D, B>
where
    GLWESwitchingKeyPrepared<D, B>: GGLWEPreparedToRef<B>,
{
    fn to_mut(&mut self) -> GGLWEPrepared<&mut [u8], B> {
        self.0.to_mut()
    }
}

impl<D: DataMut, B: Backend> GLWESwitchingKeyDegreesMut for GLWEToLWEKeyPrepared<D, B> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: DataRef, B: Backend> GLWESwitchingKeyDegrees for GLWEToLWEKeyPrepared<D, B> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}

pub type GLWEToLWEKeyPreparedBackendRef<'a, B> = GLWEToLWEKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEToLWEKeyPreparedBackendMut<'a, B> = GLWEToLWEKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWEToLWEKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEToLWEKeyPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWEToLWEKeyPreparedToBackendRef<B> for GLWEToLWEKeyPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWEToLWEKeyPreparedBackendRef<'_, B> {
        GLWEToLWEKeyPrepared(GLWESwitchingKeyPreparedToBackendRef::to_backend_ref(&self.0))
    }
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for GLWEToLWEKeyPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        self.0.key.to_backend_ref()
    }
}

pub trait GLWEToLWEKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWEToLWEKeyPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWEToLWEKeyPreparedToBackendMut<B> for GLWEToLWEKeyPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GLWEToLWEKeyPreparedBackendMut<'_, B> {
        GLWEToLWEKeyPrepared(GLWESwitchingKeyPreparedToBackendMut::to_backend_mut(&mut self.0))
    }
}
