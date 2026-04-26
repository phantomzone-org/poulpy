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

use poulpy_hal::{api::ModuleLogN, layouts::{Backend, CyclotomicOrder, GaloisElement, Module, ScratchArena, galois_element}};

pub use crate::api::GLWETrace;
use crate::{
    GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWE, GLWEAutomorphismKeyHelper, GLWEBackendMut, GLWEInfos, GLWELayout, GLWEToBackendMut,
        GLWEToBackendRef, GetGaloisElement, LWEInfos, glwe_backend_mut_from_mut, glwe_backend_ref_from_mut,
        prepared::GGLWEPreparedToBackendRef,
    },
};

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

fn trace_inplace_internal<'s, 'r, M, K, H, BE: Backend + 's>(
    module: &M,
    res: &mut GLWEBackendMut<'r, BE>,
    skip: usize,
    keys: &H,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: ModuleLogN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + GLWECopy<BE>
        + CyclotomicOrder
        + GLWENormalize<BE>
        + GLWETraceDefault<BE>
        + ?Sized,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let ksk_infos: &GGLWELayout = &keys.automorphism_key_infos();
    let log_n: usize = module.log_n();

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(ksk_infos.n(), module.n() as u32);
    assert!(skip <= log_n);
    assert_eq!(ksk_infos.rank_in(), res.rank());
    assert_eq!(ksk_infos.rank_out(), res.rank());
    assert!(
        scratch.available() >= module.glwe_trace_tmp_bytes_default(res, res, ksk_infos),
        "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
        scratch.available(),
        module.glwe_trace_tmp_bytes_default(res, res, ksk_infos)
    );

    if res.base2k() != ksk_infos.base2k() {
        let res_conv_layout = GLWELayout {
            n: module.n().into(),
            base2k: ksk_infos.base2k(),
            k: res.max_k(),
            rank: res.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut res_conv, scratch_1) = scratch_local.take_glwe(&res_conv_layout);
        let mut scratch_1 = scratch_1;

        scratch_1 = scratch_1.apply_mut(|scratch| {
            module.glwe_normalize(&mut res_conv, &glwe_backend_ref_from_mut::<BE>(&*res), scratch);
        });

        {
            let mut res_conv_backend = glwe_backend_mut_from_mut::<BE>(&mut res_conv);
            scratch_1 = scratch_1.apply_mut(|scratch| {
                trace_inplace_internal::<M, K, H, BE>(module, &mut res_conv_backend, skip, keys, scratch);
            });
        }

        scratch_1.apply_mut(|scratch| {
            module.glwe_normalize(res, &glwe_backend_ref_from_mut::<BE>(&res_conv), scratch);
        });
        return;
    }

    for i in skip..log_n {
        let p: i64 = if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) };
        let mut res_backend = &mut *res;
        module.glwe_rsh(1, &mut res_backend, scratch);
        if let Some(key) = keys.get_automorphism_key(p) {
            module.glwe_automorphism_add_inplace(res, key, scratch);
        } else {
            panic!("keys[{p}] is empty")
        }
    }
}

#[doc(hidden)]
pub trait GLWETraceDefault<BE: Backend>
where
    Self: ModuleLogN + GaloisElement + GLWEAutomorphism<BE> + GLWEShift<BE> + GLWECopy<BE> + CyclotomicOrder + GLWENormalize<BE>,
{
    fn glwe_trace_galois_elements_default(&self) -> Vec<i64> {
        trace_galois_elements(self.log_n(), self.cyclotomic_order())
    }

    fn glwe_trace_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
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
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a_infos.n(),
                base2k: key_infos.base2k(),
                k: a_infos.max_k(),
                rank: a_infos.rank(),
            };
            let lvl_1: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos) + self.glwe_normalize_tmp_bytes();
            let lvl_2: usize = self.glwe_trace_tmp_bytes_default(&a_conv_infos, &a_conv_infos, key_infos);
            return lvl_1 + lvl_2;
        }

        let lvl_1: usize = if res_infos.max_k() > a_infos.max_k() {
            GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
        } else {
            GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos)
        };

        lvl_0 + lvl_1
    }

    fn glwe_trace_default<'s, R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &'s mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        let atk_layout: &GGLWELayout = &keys.automorphism_key_infos();
        assert!(
            scratch.available() >= self.glwe_trace_tmp_bytes_default(res, a, atk_layout),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            self.glwe_trace_tmp_bytes_default(res, a, atk_layout)
        );

        let scratch_local = scratch.borrow();
        let (mut tmp, scratch_1) = scratch_local.take_glwe(&GLWELayout {
            n: res.n(),
            base2k: atk_layout.base2k(),
            k: a.max_k().max(res.max_k()),
            rank: res.rank(),
        });
        let mut scratch_1 = scratch_1;

        if a.base2k() == atk_layout.base2k() {
            self.glwe_copy(&mut glwe_backend_mut_from_mut::<BE>(&mut tmp), &a.to_backend_ref());
        } else {
            let mut tmp_backend = glwe_backend_mut_from_mut::<BE>(&mut tmp);
            scratch_1 = scratch_1.apply_mut(|scratch| {
                self.glwe_normalize(&mut tmp_backend, &a.to_backend_ref(), scratch);
            });
        }

        {
            let mut tmp_backend = glwe_backend_mut_from_mut::<BE>(&mut tmp);
            scratch_1 = scratch_1.apply_mut(|scratch| {
                trace_inplace_internal::<Self, K, H, BE>(self, &mut tmp_backend, skip, keys, scratch);
            });
        }

        if res.base2k() == atk_layout.base2k() {
            self.glwe_copy(&mut res.to_backend_mut(), &glwe_backend_ref_from_mut::<BE>(&tmp));
        } else {
            let (mut res_out, scratch_2) = scratch_1.take_glwe(&res.glwe_layout());
            {
                let mut res_out_backend = glwe_backend_mut_from_mut::<BE>(&mut res_out);
                scratch_2.apply_mut(|scratch| {
                    self.glwe_normalize(&mut res_out_backend, &glwe_backend_ref_from_mut::<BE>(&tmp), scratch);
                });
            }
            self.glwe_copy(&mut res.to_backend_mut(), &glwe_backend_ref_from_mut::<BE>(&res_out));
        }
    }

    fn glwe_trace_inplace_default<'s, R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        {
            let mut res_backend = res.to_backend_mut();
            trace_inplace_internal::<Self, K, H, BE>(self, &mut res_backend, skip, keys, scratch)
        };
    }
}

impl<BE: Backend> GLWETraceDefault<BE> for Module<BE> where
    Self: ModuleLogN + GaloisElement + GLWEAutomorphism<BE> + GLWEShift<BE> + GLWECopy<BE> + CyclotomicOrder + GLWENormalize<BE>
{
}
