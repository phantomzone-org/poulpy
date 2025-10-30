use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, GaloisElement, Module, Scratch},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToMut, GLWEToRef, GetGaloisElement},
};
pub trait GLWEPacking<BE: Backend> {
    /// Packs [x_0: GLWE(m_0), x_1: GLWE(m_1), ..., x_i: GLWE(m_i)]
    /// to [0: GLWE(m_0 * X^x_0 + m_1 * X^x_1 + ... + m_i * X^x_i)]
    fn glwe_pack<R, K, H>(&self, cts: &mut HashMap<usize, &mut R>, log_gap_out: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

impl<BE: Backend> GLWEPacking<BE> for Module<BE>
where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    /// Packs [x_0: GLWE(m_0), x_1: GLWE(m_1), ..., x_i: GLWE(m_i)]
    /// to [0: GLWE(m_0 * X^x_0 + m_1 * X^x_1 + ... + m_i * X^x_i)]
    fn glwe_pack<R, K, H>(&self, cts: &mut HashMap<usize, &mut R>, log_gap_out: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(*cts.keys().max().unwrap() < self.n())
        }

        let log_n: usize = self.log_n();

        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let key: &K = if i == 0 {
                keys.get_automorphism_key(-1).unwrap()
            } else {
                keys.get_automorphism_key(self.galois_element(1 << (i - 1)))
                    .unwrap()
            };

            for j in 0..t {
                let mut a: Option<&mut R> = cts.remove(&j);
                let mut b: Option<&mut R> = cts.remove(&(j + t));

                pack_internal(self, &mut a, &mut b, i, key, scratch);

                if let Some(a) = a {
                    cts.insert(j, a);
                } else if let Some(b) = b {
                    cts.insert(j, b);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_internal<M, A, B, K, BE: Backend>(
    module: &M,
    a: &mut Option<&mut A>,
    b: &mut Option<&mut B>,
    i: usize,
    auto_key: &K,
    scratch: &mut Scratch<BE>,
) where
    M: GLWEAutomorphism<BE> + GLWERotate<BE> + GLWESub + GLWEShift<BE> + GLWEAdd + GLWENormalize<BE>,
    A: GLWEToMut + GLWEToRef + GLWEInfos,
    B: GLWEToMut + GLWEToRef + GLWEInfos,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    // Different cases for wether a and/or b are zero.
    //
    // Implicite RSH without modulus switch, introduces extra I(X) * Q/2 on decryption.
    // Necessary so that the scaling of the plaintext remains constant.
    // It however is ok to do so here because coefficients are eventually
    // either mapped to garbage or twice their value which vanishes I(X)
    // since 2*(I(X) * Q/2) = I(X) * Q = 0 mod Q.
    if let Some(a) = a.as_deref_mut() {
        let t: i64 = 1 << (a.n().log2() - i - 1);

        if let Some(b) = b.as_deref_mut() {
            let (mut tmp_b, scratch_1) = scratch.take_glwe(a);

            // a = a * X^-t
            module.glwe_rotate_inplace(-t, a, scratch_1);

            // tmp_b = a * X^-t - b
            module.glwe_sub(&mut tmp_b, a, b);
            module.glwe_rsh(1, &mut tmp_b, scratch_1);

            // a = a * X^-t + b
            module.glwe_add_inplace(a, b);
            module.glwe_rsh(1, a, scratch_1);

            module.glwe_normalize_inplace(&mut tmp_b, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            module.glwe_automorphism_inplace(&mut tmp_b, auto_key, scratch_1);

            // a = a * X^-t + b - phi(a * X^-t - b)
            module.glwe_sub_inplace(a, &tmp_b);
            module.glwe_normalize_inplace(a, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            module.glwe_rotate_inplace(t, a, scratch_1);
        } else {
            module.glwe_rsh(1, a, scratch);
            // a = a + phi(a)
            module.glwe_automorphism_add_inplace(a, auto_key, scratch);
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let (mut tmp_b, scratch_1) = scratch.take_glwe(b);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        module.glwe_automorphism_sub_negate(b, &tmp_b, auto_key, scratch_1);
    }
}
