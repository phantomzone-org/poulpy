use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch},
};

use crate::{
    GGSWExpandRows, ScratchTakeCore,
    automorphism::glwe_ct::GLWEAutomorphism,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedToRef, GGSW, GGSWInfos, GGSWToMut,
        GGSWToRef, GetGaloisElement,
    },
};

impl GGSW<Vec<u8>> {
    pub fn automorphism_tmp_bytes<R, A, K, T, M, BE: Backend>(
        module: &M,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
        M: GGSWAutomorphism<BE>,
    {
        module.ggsw_automorphism_tmp_bytes(res_infos, a_infos, key_infos, tsk_infos)
    }
}

impl<D: DataMut> GGSW<D> {
    pub fn automorphism<A, K, T, M, BE: Backend>(&mut self, module: &M, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        A: GGSWToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGSWAutomorphism<BE>,
    {
        module.ggsw_automorphism(self, a, key, tsk, scratch);
    }

    pub fn automorphism_inplace<K, T, M, BE: Backend>(&mut self, module: &M, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGSWAutomorphism<BE>,
    {
        module.ggsw_automorphism_inplace(self, key, tsk, scratch);
    }
}

impl<BE: Backend> GGSWAutomorphism<BE> for Module<BE> where Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE> {}

pub trait GGSWAutomorphism<BE: Backend>
where
    Self: GLWEAutomorphism<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
            .max(self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
    }

    fn ggsw_automorphism<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        assert_eq!(res.dsize(), a.dsize());
        assert!(res.dnum() <= a.dnum());
        assert!(scratch.available() >= self.ggsw_automorphism_tmp_bytes(res, a, key, tsk));

        // Keyswitch the j-th row of the col 0
        for row in 0..res.dnum().as_usize() {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.glwe_automorphism(&mut res.at_mut(row, 0), &a.at(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }

    fn ggsw_automorphism_inplace<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let tsk: &GGLWEToGGSWKeyPrepared<&[u8], BE> = &tsk.to_ref();

        // Keyswitch the j-th row of the col 0
        for row in 0..res.dnum().as_usize() {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            self.glwe_automorphism_inplace(&mut res.at_mut(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}

impl<DataSelf: DataMut> GGSW<DataSelf> {}
