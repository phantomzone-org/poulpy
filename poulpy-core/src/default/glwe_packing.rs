use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, GaloisElement, Module, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETrace, ScratchArenaTakeCore,
    layouts::{
        BackendGLWE, GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement, ModuleCoreAlloc, prepared::GGLWEPreparedToBackendRef,
    },
};

fn glwe_rotate_assign_on<'s, M, A, BE: Backend + 's>(module: &M, k: i64, a: &mut A, scratch: &mut ScratchArena<'s, BE>)
where
    M: GLWERotate<BE> + ?Sized,
    A: GLWEToBackendMut<BE>,
{
    let mut a_backend = a.to_backend_mut();
    module.glwe_rotate_assign(k, &mut a_backend, scratch);
}

fn glwe_normalize_assign_on<'s, M, A, BE: Backend + 's>(module: &M, a: &mut A, scratch: &mut ScratchArena<'s, BE>)
where
    M: GLWENormalize<BE> + ?Sized,
    A: GLWEToBackendMut<BE>,
{
    let mut a_backend = a.to_backend_mut();
    module.glwe_normalize_assign(&mut a_backend, scratch)
}

fn glwe_automorphism_add_assign_on<'s, M, A, K, BE: Backend + 's>(
    module: &M,
    a: &mut A,
    key: &K,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: GLWEAutomorphism<BE> + ?Sized,
    A: GLWEToBackendMut<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
{
    module.glwe_automorphism_add_assign(a, key, scratch)
}

#[allow(clippy::too_many_arguments)]
fn pack_internal<'s, M, A, B, K, BE: Backend + 's>(
    module: &M,
    a: &mut Option<&mut A>,
    b: &mut Option<&mut B>,
    i: usize,
    auto_key: &K,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: GLWEAutomorphism<BE>
        + GLWERotate<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ?Sized,
    A: GLWEToBackendMut<BE> + GLWEInfos,
    B: GLWEToBackendMut<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    if let Some(a) = a.as_deref_mut() {
        let t: i64 = 1 << (a.n().log2() - i - 1);

        if let Some(b) = b.as_deref_mut() {
            let a_layout = a.glwe_layout();
            let mut tmp_b = module.glwe_alloc_from_infos(&a_layout);

            glwe_rotate_assign_on(module, -t, a, scratch);

            module.glwe_sub(&mut tmp_b, a, b);
            module.glwe_rsh(1, &mut tmp_b, scratch);

            {
                let mut a_backend = a.to_backend_mut();
                let b_backend = b.to_backend_ref();
                module.glwe_add_assign_backend(&mut a_backend, &b_backend);
            }
            module.glwe_rsh(1, a, scratch);

            {
                let mut tmp_b_backend: GLWEBackendMut<'_, BE> =
                    <BackendGLWE<BE> as GLWEToBackendMut<BE>>::to_backend_mut(&mut tmp_b);
                module.glwe_normalize_assign(&mut tmp_b_backend, scratch);
            }

            module.glwe_automorphism_assign(&mut tmp_b, auto_key, scratch);

            module.glwe_sub_assign(a, &tmp_b);
            glwe_normalize_assign_on(module, a, scratch);

            glwe_rotate_assign_on(module, t, a, scratch)
        } else {
            module.glwe_rsh(1, a, scratch);
            glwe_automorphism_add_assign_on(module, a, auto_key, scratch);
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let b_layout = b.glwe_layout();
        let mut tmp_b = module.glwe_alloc_from_infos(&b_layout);
        {
            let b_backend = b.to_backend_ref();
            let mut tmp_b_backend: GLWEBackendMut<'_, BE> = <BackendGLWE<BE> as GLWEToBackendMut<BE>>::to_backend_mut(&mut tmp_b);
            module.glwe_rotate(t, &mut tmp_b_backend, &b_backend);
        }
        module.glwe_rsh(1, &mut tmp_b, scratch);

        module.glwe_automorphism_sub_negate(b, &tmp_b, auto_key, scratch)
    }
}

#[doc(hidden)]
pub(crate) trait GLWEPackingDefault<BE: Backend>
where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + GLWERotate<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>
        + GLWETrace<BE>,
{
    fn glwe_pack_galois_elements_default(&self) -> Vec<i64> {
        self.glwe_trace_galois_elements()
    }

    fn glwe_pack_tmp_bytes_default<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, key.n());

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        let lvl_1: usize = self
            .glwe_rotate_tmp_bytes()
            .max(self.glwe_shift_tmp_bytes())
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.glwe_automorphism_tmp_bytes(res, res, key));

        (lvl_0 + lvl_1).max(self.glwe_trace_tmp_bytes(res, res, key))
    }

    fn glwe_pack_default<'s, R, A, K, H>(
        &self,
        res: &mut R,
        mut a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        BE: 's,
    {
        assert!(*a.keys().max().unwrap() < self.n());
        let key_infos = keys.automorphism_key_infos();
        assert!(
            scratch.available() >= self.glwe_pack_tmp_bytes_default(res, &key_infos),
            "scratch.available(): {} < GLWEPacking::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            self.glwe_pack_tmp_bytes_default(res, &key_infos)
        );

        let mut scratch_local = scratch.borrow();
        let log_n: usize = self.log_n();
        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let key: &K = if i == 0 {
                keys.get_automorphism_key(-1).unwrap()
            } else {
                keys.get_automorphism_key(self.galois_element(1 << (i - 1))).unwrap()
            };

            for j in 0..t {
                let mut lo: Option<&mut A> = a.remove(&j);
                let mut hi: Option<&mut A> = a.remove(&(j + t));

                scratch_local = scratch_local.apply_mut(|scratch| {
                    pack_internal(self, &mut lo, &mut hi, i, key, scratch);
                });

                if let Some(lo) = lo {
                    a.insert(j, lo);
                } else if let Some(hi) = hi {
                    a.insert(j, hi);
                }
            }
        }

        scratch_local.apply_mut(|scratch| {
            self.glwe_trace(res, log_n - log_gap_out, *a.get_mut(&0).unwrap(), keys, scratch);
        });
    }
}

impl<BE: Backend> GLWEPackingDefault<BE> for Module<BE> where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + GLWERotate<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>
        + GLWETrace<BE>
{
}
