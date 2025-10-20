use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch, VecZnx};

use crate::{
    GGSWExpandRows, ScratchTakeCore,
    keyswitching::glwe_ct::GLWEKeyswitch,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GGSW, GGSWInfos, GGSWToMut, GGSWToRef, prepared::TensorKeyPreparedToRef},
};

impl GGSW<Vec<u8>> {
    pub fn keyswitch_tmp_bytes<R, A, K, T, M, BE: Backend>(
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
        M: GGSWKeyswitch<BE>,
    {
        module.ggsw_keyswitch_tmp_bytes(res_infos, a_infos, key_infos, tsk_infos)
    }
}

impl<D: DataMut> GGSW<D> {
    pub fn keyswitch<M, A, K, T, BE: Backend>(&mut self, module: &M, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGSWKeyswitch<BE>,
    {
        module.ggsw_keyswitch(self, a, key, tsk, scratch);
    }

    pub fn keyswitch_inplace<M, K, T, BE: Backend>(&mut self, module: &M, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        K: GGLWEPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GGSWKeyswitch<BE>,
    {
        module.ggsw_keyswitch_inplace(self, key, tsk, scratch);
    }
}

impl<BE: Backend> GGSWKeyswitch<BE> for Module<BE> where Self: GLWEKeyswitch<BE> + GGSWExpandRows<BE> {}

pub trait GGSWKeyswitch<BE: Backend>
where
    Self: GLWEKeyswitch<BE> + GGSWExpandRows<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        assert_eq!(key_infos.rank_in(), key_infos.rank_out());
        assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
        assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());

        let rank: usize = key_infos.rank_out().into();

        let size_out: usize = res_infos.k().div_ceil(res_infos.base2k()) as usize;
        let res_znx: usize = VecZnx::bytes_of(self.n(), rank + 1, size_out);
        let ci_dft: usize = self.bytes_of_vec_znx_dft(rank + 1, size_out);
        let ks: usize = self.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
        let expand_rows: usize = self.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos);
        let res_dft: usize = self.bytes_of_vec_znx_dft(rank + 1, size_out);

        if a_infos.base2k() == tsk_infos.base2k() {
            res_znx + ci_dft + (ks | expand_rows | res_dft)
        } else {
            let a_conv: usize = VecZnx::bytes_of(
                self.n(),
                1,
                res_infos.k().div_ceil(tsk_infos.base2k()) as usize,
            ) + self.vec_znx_normalize_tmp_bytes();
            res_znx + ci_dft + (a_conv | ks | expand_rows | res_dft)
        }
    }

    fn ggsw_keyswitch<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for row in 0..a.dnum().into() {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.glwe_keyswitch(&mut res.at_mut(row, 0), &a.at(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }

    fn ggsw_keyswitch_inplace<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE>,
        T: TensorKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        for row in 0..res.dnum().into() {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            self.glwe_keyswitch_inplace(&mut res.at_mut(row, 0), key, scratch);
        }

        self.ggsw_expand_row(res, tsk, scratch);
    }
}
