use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataMut, ScratchArena, ZnxView, ZnxViewMut, ZnxZero},
};

use crate::{
    ScratchArenaTakeCore,
    api::{GLWEKeyswitch, GLWERotate},
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWBackendMut, GGSWInfos, GGSWToBackendMut, GGSWToMut, GLWE, GLWEInfos, GLWELayout,
        GLWEToBackendRef, GLWEToRef, LWE, LWEInfos, LWEToMut, Rank, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

pub trait LWESampleExtract
where
    Self: ModuleN,
{
    fn lwe_sample_extract<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToMut,
        A: GLWEToRef,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert!(res.n() <= a.n());
        assert_eq!(a.n(), self.n() as u32);
        assert!(res.base2k() == a.base2k());

        let min_size: usize = res.size().min(a.size());
        let n: usize = res.n().into();

        res.data.zero();
        (0..min_size).for_each(|i| {
            let data_lwe: &mut [i64] = res.data.at_mut(0, i);
            data_lwe[0] = a.data.at(0, i)[0];
            data_lwe[1..].copy_from_slice(&a.data.at(1, i)[..n]);
        });
    }
}

pub trait GLWEFromLWE<BE: Backend>
where
    Self: GLWEKeyswitch<BE>,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToMut + crate::layouts::GLWEToBackendMut<BE>,
        A: crate::layouts::LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

pub trait LWEFromGLWE<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + LWESampleExtract + GLWERotate<BE>,
{
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, glwe_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let res_infos: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: lwe_infos.base2k(),
            k: lwe_infos.max_k(),
            rank: Rank(1),
        };

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of(self.n().into(), lwe_infos.base2k(), lwe_infos.max_k(), 1u32.into());
        let lvl_1: usize = self.glwe_keyswitch_tmp_bytes(&res_infos, glwe_infos, key_infos);
        let lvl_2: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos);

        lvl_0 + lvl_1 + lvl_2
    }

    fn lwe_from_glwe<'s, R, A, K>(&self, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToMut,
        A: GLWEToRef + GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let a_ref: &GLWE<&[u8]> = &a.to_ref();
        let a_backend = a.to_backend_ref();

        assert_eq!(a_ref.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);
        assert!(res.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_from_glwe_tmp_bytes(res, a_ref, key),
            "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
            scratch.available(),
            self.lwe_from_glwe_tmp_bytes(res, a_ref, key)
        );

        let glwe_layout: GLWELayout = GLWELayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: res.max_k(),
            rank: Rank(1),
        };

        let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.borrow().take_glwe(&glwe_layout);

        match a_idx {
            0 => self.glwe_keyswitch(&mut tmp_glwe_rank_1, &a_backend, key, &mut scratch_1),
            _ => {
                let (mut tmp_glwe_in, mut scratch_2) = scratch_1.take_glwe(&a_backend);
                self.glwe_rotate(-(a_idx as i64), &mut tmp_glwe_in, &a_backend);
                let tmp_glwe_in_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_in);
                self.glwe_keyswitch(&mut tmp_glwe_rank_1, &tmp_glwe_in_ref, key, &mut scratch_2)
            }
        }

        self.lwe_sample_extract(res, &tmp_glwe_rank_1);
    }
}

pub trait GGSWFromGGLWE<BE: Backend> {
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut;
}

pub trait GGSWExpandRows<BE: Backend> {
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, 'r, T>(&self, res: &mut GGSWBackendMut<'r, BE>, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}
