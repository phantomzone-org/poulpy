use poulpy_hal::{
    api::{ModuleLogN, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, CyclotomicOrder, DataMut, GaloisElement, Module, Scratch, VecZnx, galois_element},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWEShift, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWELayout, GLWEToMut,
        GLWEToRef, GetGaloisElement, LWEInfos,
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
        module.glwe_trace_tmp_bytes(res_infos, a_infos, key_infos)
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn trace<A, H, K, M, BE: Backend>(&mut self, module: &M, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWETrace<BE>,
    {
        module.glwe_trace(self, skip, a, keys, scratch);
    }

    pub fn trace_inplace<H, K, M, BE: Backend>(&mut self, module: &M, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
        M: GLWETrace<BE>,
    {
        module.glwe_trace_inplace(self, skip, keys, scratch);
    }
}

#[inline(always)]
pub fn trace_galois_elements(log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
    (0..log_n)
        .map(|i| {
            if i == 0 {
                -1
            } else {
                galois_element(1 << (i - 1), cyclotomic_order)
            }
        })
        .collect()
}

impl<BE: Backend> GLWETrace<BE> for Module<BE>
where
    Self: ModuleLogN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + GLWECopy
        + CyclotomicOrder
        + VecZnxNormalizeTmpBytes
        + VecZnxNormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_trace_galois_elements(&self) -> Vec<i64> {
        trace_galois_elements(self.log_n(), self.cyclotomic_order())
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

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        self.glwe_copy(res, a);
        self.glwe_trace_inplace(res, skip, keys, scratch);
    }

    fn glwe_trace_inplace<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let ksk_infos: &GGLWELayout = &keys.automorphism_key_infos();
        let log_n: usize = self.log_n();

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(ksk_infos.n(), self.n() as u32);
        assert!(skip <= log_n);
        assert_eq!(ksk_infos.rank_in(), res.rank());
        assert_eq!(ksk_infos.rank_out(), res.rank());

        if res.base2k() != ksk_infos.base2k() {
            let (mut self_conv, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: self.n().into(),
                base2k: ksk_infos.base2k(),
                k: res.k(),
                rank: res.rank(),
            });

            for j in 0..(res.rank() + 1).into() {
                self.vec_znx_normalize(
                    ksk_infos.base2k().into(),
                    &mut self_conv.data,
                    j,
                    res.base2k().into(),
                    res.data(),
                    j,
                    scratch_1,
                );
            }

            for i in skip..log_n {
                self.glwe_rsh(1, &mut self_conv, scratch_1);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    self.galois_element(1 << (i - 1))
                };

                if let Some(key) = keys.get_automorphism_key(p) {
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
                    ksk_infos.base2k().into(),
                    &self_conv.data,
                    j,
                    scratch_1,
                );
            }
        } else {
            // println!("res: {}", res);

            for i in skip..log_n {
                self.glwe_rsh(1, res, scratch);

                let p: i64 = if i == 0 {
                    -1
                } else {
                    self.galois_element(1 << (i - 1))
                };

                if let Some(key) = keys.get_automorphism_key(p) {
                    self.glwe_automorphism_add_inplace(res, key, scratch);
                } else {
                    panic!("keys[{p}] is empty")
                }
            }
        }
    }
}

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_trace_inplace<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
