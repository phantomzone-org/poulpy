//! GLWE trace operation (sum of Galois automorphisms).
//!
//! The trace maps a GLWE ciphertext encrypting a polynomial `m(X)` to one
//! encrypting the sum of its Galois conjugates:
//!
//! `Trace(ct) = sum_{i in S} phi_i(ct)`
//!
//! where `phi_i` are the Galois automorphisms `X -> X^{g^i}`.
//! This is the dual operation of slot packing: it projects a ciphertext
//! onto a smaller subspace of plaintext slots, effectively replicating
//! a single slot value across multiple positions.
//!
//! The `skip` parameter controls how many initial automorphism levels
//! are skipped, allowing partial traces that project onto larger subspaces.
//!
//! Requires automorphism keys indexed by the Galois elements returned
//! from [`GLWETrace::glwe_trace_galois_elements`].

use poulpy_hal::{
    api::{ModuleLogN, ScratchAvailable, VecZnxNormalizeTmpBytes},
    layouts::{Backend, CyclotomicOrder, DataMut, GaloisElement, Module, Scratch, VecZnx, galois_element},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWELayout, GLWEToMut,
        GLWEToRef, GetGaloisElement, LWEInfos,
    },
};

impl GLWE<Vec<u8>> {
    /// Returns the Galois elements required by the trace operation.
    ///
    /// The caller must provide automorphism keys for each of these
    /// elements before invoking [`GLWE::trace`] or [`GLWE::trace_inplace`].
    pub fn trace_galois_elements<M, BE: Backend>(module: &M) -> Vec<i64>
    where
        M: GLWETrace<BE>,
    {
        module.glwe_trace_galois_elements()
    }

    /// Returns the number of scratch bytes required by [`GLWE::trace`].
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
        A: GLWEToRef + GLWEInfos,
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
        + GLWENormalize<BE>,
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
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let lvl_0: usize = self.glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos);
        if a_infos.base2k() != key_infos.base2k() {
            let lvl_1: usize = VecZnx::bytes_of(
                self.n(),
                (key_infos.rank_out() + 1).into(),
                res_infos.k().min(a_infos.k()).div_ceil(key_infos.base2k()) as usize,
            ) + self.vec_znx_normalize_tmp_bytes();
            return lvl_0 + lvl_1;
        }

        let lvl_1: usize = if res_infos.k() > a_infos.k() {
            GLWE::bytes_of_from_infos(res_infos)
        } else {
            GLWE::bytes_of_from_infos(a_infos)
        };

        lvl_0 + lvl_1
    }

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let atk_layout: &GGLWELayout = &keys.automorphism_key_infos();
        assert!(
            scratch.available() >= self.glwe_trace_tmp_bytes(res, a, atk_layout),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            self.glwe_trace_tmp_bytes(res, a, atk_layout)
        );

        let (mut tmp, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: res.n(),
            base2k: atk_layout.base2k(),
            k: a.k().max(res.k()),
            rank: res.rank(),
        });

        if a.base2k() == atk_layout.base2k() {
            self.glwe_copy(&mut tmp, a);
        } else {
            self.glwe_normalize(&mut tmp, a, scratch_1);
        }

        self.glwe_trace_inplace(&mut tmp, skip, keys, scratch_1);

        if res.base2k() == atk_layout.base2k() {
            self.glwe_copy(res, &tmp);
        } else {
            self.glwe_normalize(res, &tmp, scratch_1);
        }
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
        assert!(
            scratch.available() >= self.glwe_trace_tmp_bytes(res, res, ksk_infos),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            self.glwe_trace_tmp_bytes(res, res, ksk_infos)
        );

        if res.base2k() != ksk_infos.base2k() {
            let (mut res_conv, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: self.n().into(),
                base2k: ksk_infos.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_1);
            self.glwe_trace_inplace(&mut res_conv, skip, keys, scratch_1);
            self.glwe_normalize(res, &res_conv, scratch_1);
        } else {
            for i in skip..log_n {
                self.glwe_rsh(1, res, scratch);

                let p: i64 = if i == 0 { -1 } else { self.galois_element(1 << (i - 1)) };

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
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_trace_inplace<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
