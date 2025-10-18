use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch},
};

use crate::{
    GGSWExpandRows, ScratchTakeCore,
    automorphism::glwe_ct::GLWEAutomorphism,
    layouts::{
        GGLWEInfos, GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWEInfos, LWEInfos,
        prepared::{AutomorphismKeyPrepared, AutomorphismKeyPreparedToRef, TensorKeyPrepared, TensorKeyPreparedToRef},
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
        K: AutomorphismKeyPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGSWAutomorphism<BE>,
    {
        module.ggsw_automorphism(self, a, key, tsk, scratch);
    }

    pub fn automorphism_inplace<K, T, M, BE: Backend>(&mut self, module: &M, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        K: AutomorphismKeyPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
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
        let out_size: usize = res_infos.size();
        let ci_dft: usize = self.bytes_of_vec_znx_dft((key_infos.rank_out() + 1).into(), out_size);
        let ks_internal: usize = self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos);
        let expand: usize = self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos);
        ci_dft + (ks_internal.max(expand))
    }

    fn ggsw_automorphism<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: AutomorphismKeyPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();
        let key: &AutomorphismKeyPrepared<&[u8], BE> = &key.to_ref();
        let tsk: &TensorKeyPrepared<&[u8], BE> = &tsk.to_ref();

        assert_eq!(res.ggsw_layout(), a.ggsw_layout());
        assert_eq!(res.glwe_layout(), a.glwe_layout());
        assert_eq!(res.lwe_layout(), a.lwe_layout());
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
        K: AutomorphismKeyPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let key: &AutomorphismKeyPrepared<&[u8], BE> = &key.to_ref();
        let tsk: &TensorKeyPrepared<&[u8], BE> = &tsk.to_ref();

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
