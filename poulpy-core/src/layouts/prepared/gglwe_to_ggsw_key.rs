use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEPreparedToMut, GGLWEPreparedToRef,
    GGLWEToGGSWKey, GGLWEToGGSWKeyToRef, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};

pub struct GGLWEToGGSWKeyPrepared<D: Data, BE: Backend> {
    pub(crate) keys: Vec<GGLWEPrepared<D, BE>>,
}

impl<D: Data, BE: Backend> LWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
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

impl<D: Data, BE: Backend> GLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data, BE: Backend> GGLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
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

pub trait GGLWEToGGSWKeyPreparedFactory<BE: Backend> {
    fn alloc_gglwe_to_ggsw_key_prepared_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>
    where
        A: GGLWEInfos;

    fn alloc_gglwe_to_ggsw_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>;

    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize;

    fn prepare_gglwe_to_ggsw_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn prepare_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToGGSWKeyPreparedToMut<BE>,
        O: GGLWEToGGSWKeyToRef;
}

impl<BE: Backend> GGLWEToGGSWKeyPreparedFactory<BE> for Module<BE>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn alloc_gglwe_to_ggsw_key_prepared_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.alloc_gglwe_to_ggsw_key_prepared(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn alloc_gglwe_to_ggsw_key_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEToGGSWKeyPrepared<Vec<u8>, BE> {
        GGLWEToGGSWKeyPrepared {
            keys: (0..rank.as_usize())
                .map(|_| self.alloc_gglwe_prepared(base2k, k, rank, rank, dnum, dsize))
                .collect(),
        }
    }

    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.bytes_of_gglwe_to_ggsw(infos.base2k(), infos.k(), infos.rank(), infos.dnum(), infos.dsize())
    }

    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        rank.as_usize() * self.bytes_of_gglwe_prepared(base2k, k, rank, rank, dnum, dsize)
    }

    fn prepare_gglwe_to_ggsw_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.prepare_gglwe_tmp_bytes(infos)
    }

    fn prepare_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToGGSWKeyPreparedToMut<BE>,
        O: GGLWEToGGSWKeyToRef,
    {
        let res: &mut GGLWEToGGSWKeyPrepared<&mut [u8], BE> = &mut res.to_mut();
        let other: &GGLWEToGGSWKey<&[u8]> = &other.to_ref();

        assert_eq!(res.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.prepare_gglwe(a, b, scratch);
        }
    }
}

impl<BE: Backend> GGLWEToGGSWKeyPrepared<Vec<u8>, BE> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.alloc_gglwe_to_ggsw_key_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.alloc_gglwe_to_ggsw_key_prepared(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.bytes_of_gglwe_to_ggsw_from_infos(infos)
    }

    pub fn bytes_of<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
    {
        module.bytes_of_gglwe_to_ggsw(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    pub fn prepare<M, O>(&mut self, module: &M, other: &O, scratch: &mut Scratch<BE>)
    where
        M: GGLWEToGGSWKeyPreparedFactory<BE>,
        O: GGLWEToGGSWKeyToRef,
    {
        module.prepare_gglwe_to_ggsw_key(self, other, scratch);
    }
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    // Returns a mutable reference to GGLWEPrepared_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: DataRef, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    // Returns a reference to GGLWEPrepared_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at(&self, i: usize) -> &GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

pub trait GGLWEToGGSWKeyPreparedToRef<BE: Backend> {
    fn to_ref(&self) -> GGLWEToGGSWKeyPrepared<&[u8], BE>;
}

impl<D: DataRef, BE: Backend> GGLWEToGGSWKeyPreparedToRef<BE> for GGLWEToGGSWKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToRef<BE>,
{
    fn to_ref(&self) -> GGLWEToGGSWKeyPrepared<&[u8], BE> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyPreparedToMut<BE: Backend> {
    fn to_mut(&mut self) -> GGLWEToGGSWKeyPrepared<&mut [u8], BE>;
}

impl<D: DataMut, BE: Backend> GGLWEToGGSWKeyPreparedToMut<BE> for GGLWEToGGSWKeyPrepared<D, BE>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToMut<BE>,
{
    fn to_mut(&mut self) -> GGLWEToGGSWKeyPrepared<&mut [u8], BE> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
