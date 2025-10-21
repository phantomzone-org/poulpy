use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, DataMut, GaloisElement, Module, Scratch, VecZnx},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWEShift, ScratchTakeCore,
    layouts::{
        Base2K, GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos,
    },
};

impl GLWE<Vec<u8>> {
    pub fn trace_galois_elements<M, BE: Backend>(module: &M) -> Vec<i64>
    where
        M: GLWETrace<BE>,
    {
        module.glwe_trace_galois_elements()
    }

    pub fn trace_tmp_bytes<R, A, K, M, BE: Backend>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
        M: GLWETrace<BE>,
    {
        module.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn trace<A, K, M, BE: Backend>(
        &mut self,
        module: &M,
        start: usize,
        end: usize,
        a: &A,
        keys: &HashMap<i64, K>,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWETrace<BE>,
    {
        module.glwe_trace(self, start, end, a, keys, scratch);
    }

    pub fn trace_inplace<K, M, BE: Backend>(
        &mut self,
        module: &M,
        start: usize,
        end: usize,
        keys: &HashMap<i64, K>,
        scratch: &mut Scratch<BE>,
    ) where
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWETrace<BE>,
    {
        module.glwe_trace_inplace(self, start, end, keys, scratch);
    }
}

impl<BE: Backend> GLWETrace<BE> for Module<BE> where
    Self: ModuleLogN + GaloisElement + GLWEAutomorphism<BE> + GLWEShift<BE> + GLWECopy
{
}

pub trait GLWETrace<BE: Backend>
where
    Self: ModuleLogN + GaloisElement + GLWEAutomorphism<BE> + GLWEShift<BE> + GLWECopy,
{
    fn glwe_trace_galois_elements(&self) -> Vec<i64> {
        (0..self.log_n())
            .map(|i| {
                if i == 0 {
                    -1
                } else {
                    self.galois_element(1 << (i - 1))
                }
            })
            .collect()
    }

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let trace: usize = self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos);
        if a_infos.base2k() != key_infos.base2k() {
            let glwe_conv: usize = VecZnx::bytes_of(
                self.n(),
                (key_infos.rank_out() + 1).into(),
                res_infos.k().min(a_infos.k()).div_ceil(key_infos.base2k()) as usize,
            ) + self.vec_znx_normalize_tmp_bytes();
            return glwe_conv + trace;
        }

        trace
    }

    fn glwe_trace<R, A, K>(&self, res: &mut R, start: usize, end: usize, a: &A, keys: &HashMap<i64, K>, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_copy(res, a);
        self.glwe_trace_inplace(res, start, end, keys, scratch);
    }

    fn glwe_trace_inplace<R, K>(&self, res: &mut R, start: usize, end: usize, keys: &HashMap<i64, K>, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let basek_ksk: Base2K = keys.get(keys.keys().next().unwrap()).unwrap().base2k();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), self.n() as u32);
            assert!(start < end);
            assert!(end <= self.log_n());
            for key in keys.values() {
                assert_eq!(key.n(), self.n() as u32);
                assert_eq!(key.base2k(), basek_ksk);
                assert_eq!(key.rank_in(), res.rank());
                assert_eq!(key.rank_out(), res.rank());
            }
        }

        if res.base2k() != basek_ksk {
            let (mut self_conv, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: self.n().into(),
                base2k: basek_ksk,
                k: res.k(),
                rank: res.rank(),
            });

            for j in 0..(res.rank() + 1).into() {
                self.vec_znx_normalize(
                    basek_ksk.into(),
                    &mut self_conv.data,
                    j,
                    basek_ksk.into(),
                    res.data(),
                    j,
                    scratch_1,
                );
            }

            for i in start..end {
                self.glwe_rsh(1, &mut self_conv, scratch_1);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    self.galois_element(1 << (i - 1))
                };

                if let Some(key) = keys.get(&p) {
                    self.glwe_automorphism_add_inplace(&mut self_conv, key, scratch_1);
                } else {
                    panic!("keys[{p}] is empty")
                }
            }

            for j in 0..(res.rank() + 1).into() {
                self.vec_znx_normalize(
                    res.base2k().into(),
                    res.data_mut(),
                    j,
                    basek_ksk.into(),
                    &self_conv.data,
                    j,
                    scratch_1,
                );
            }
        } else {
            for i in start..end {
                self.glwe_rsh(1, res, scratch);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    self.galois_element(1 << (i - 1))
                };

                if let Some(key) = keys.get(&p) {
                    self.glwe_automorphism_add_inplace(res, key, scratch);
                } else {
                    panic!("keys[{p}] is empty")
                }
            }
        }
    }
}
