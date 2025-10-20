use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, GaloisElement, Module, Scratch},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, ScratchTakeCore,
    glwe_trace::GLWETrace,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, prepared::GetAutomorphismGaloisElement,
    },
};

/// [GLWEPacker] enables only the fly GLWE packing
/// with constant memory of Log(N) ciphertexts.
/// Main difference with usual GLWE packing is that
/// the output is bit-reversed.
pub struct GLWEPacker {
    accumulators: Vec<Accumulator>,
    log_batch: usize,
    counter: usize,
}

/// [Accumulator] stores intermediate packing result.
/// There are Log(N) such accumulators in a [GLWEPacker].
struct Accumulator {
    data: GLWE<Vec<u8>>,
    value: bool,   // Implicit flag for zero ciphertext
    control: bool, // Can be combined with incoming value
}

impl Accumulator {
    /// Allocates a new [Accumulator].
    ///
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `base2k`: base 2 logarithm of the GLWE ciphertext in memory digit representation.
    /// * `k`: base 2 precision of the GLWE ciphertext precision over the Torus.
    /// * `rank`: rank of the GLWE ciphertext.
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self {
            data: GLWE::alloc_from_infos(infos),
            value: false,
            control: false,
        }
    }
}

impl GLWEPacker {
    /// Instantiates a new [GLWEPacker].
    ///
    /// # Arguments
    ///
    /// * `log_batch`: packs coefficients which are multiples of X^{N/2^log_batch}.
    ///   i.e. with `log_batch=0` only the constant coefficient is packed
    ///   and N GLWE ciphertext can be packed. With `log_batch=2` all coefficients
    ///   which are multiples of X^{N/4} are packed. Meaning that N/4 ciphertexts
    ///   can be packed.
    pub fn alloc<A>(infos: &A, log_batch: usize) -> Self
    where
        A: GLWEInfos,
    {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = infos.n().log2();
        (0..log_n - log_batch).for_each(|_| accumulators.push(Accumulator::alloc(infos)));
        GLWEPacker {
            accumulators,
            log_batch,
            counter: 0,
        }
    }

    /// Implicit reset of the internal state (to be called before a new packing procedure).
    fn reset(&mut self) {
        for i in 0..self.accumulators.len() {
            self.accumulators[i].value = false;
            self.accumulators[i].control = false;
        }
        self.counter = 0;
    }

    /// Number of scratch space bytes required to call [Self::add].
    pub fn tmp_bytes<R, K, M, BE: Backend>(module: &M, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
        M: GLWEPacking<BE>,
    {
        GLWE::bytes_of_from_infos(res_infos)
            + module
                .glwe_rsh_tmp_byte()
                .max(module.glwe_automorphism_tmp_bytes(res_infos, res_infos, key_infos))
    }

    pub fn galois_elements<M, BE: Backend>(module: &M) -> Vec<i64>
    where
        M: GLWETrace<BE>,
    {
        module.glwe_trace_galois_elements()
    }

    /// Adds a GLWE ciphertext to the [GLWEPacker].
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `res`: space to append fully packed ciphertext. Only when the number
    ///   of packed ciphertexts reaches N/2^log_batch is a result written.
    /// * `a`: ciphertext to pack. Can optionally give None to pack a 0 ciphertext.
    /// * `auto_keys`: a [HashMap] containing the [AutomorphismKeyExec]s.
    /// * `scratch`: scratch space of size at least [Self::tmp_bytes].
    pub fn add<A, K, M, BE: Backend>(&mut self, module: &M, a: Option<&A>, auto_keys: &HashMap<i64, K>, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetAutomorphismGaloisElement + GGLWEInfos,
        M: GLWEPacking<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            (self.counter as u32) < self.accumulators[0].data.n(),
            "Packing limit of {} reached",
            self.accumulators[0].data.n().0 as usize >> self.log_batch
        );

        pack_core(
            module,
            a,
            &mut self.accumulators,
            self.log_batch,
            auto_keys,
            scratch,
        );
        self.counter += 1 << self.log_batch;
    }

    /// Flush result to`res`.
    pub fn flush<R, M, BE: Backend>(&mut self, module: &M, res: &mut R)
    where
        R: GLWEToMut,
        M: GLWEPacking<BE>,
    {
        assert!(self.counter as u32 == self.accumulators[0].data.n());
        // Copy result GLWE into res GLWE
        module.glwe_copy(
            res,
            &self.accumulators[module.log_n() - self.log_batch - 1].data,
        );

        self.reset();
    }
}

impl<BE: Backend> GLWEPacking<BE> for Module<BE> where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy
{
}

pub trait GLWEPacking<BE: Backend>
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
{
    /// Packs [x_0: GLWE(m_0), x_1: GLWE(m_1), ..., x_i: GLWE(m_i)]
    /// to [0: GLWE(m_0 * X^x_0 + m_1 * X^x_1 + ... + m_i * X^x_i)]
    fn glwe_pack<R, K>(
        &self,
        cts: &mut HashMap<usize, &mut R>,
        log_gap_out: usize,
        keys: &HashMap<i64, K>,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetAutomorphismGaloisElement + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(*cts.keys().max().unwrap() < self.n())
        }

        let log_n: usize = self.log_n();

        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let key: &K = if i == 0 {
                keys.get(&-1).unwrap()
            } else {
                keys.get(&self.galois_element(1 << (i - 1))).unwrap()
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

fn pack_core<A, K, M, BE: Backend>(
    module: &M,
    a: Option<&A>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &HashMap<i64, K>,
    scratch: &mut Scratch<BE>,
) where
    A: GLWEToRef + GLWEInfos,
    K: GGLWEPreparedToRef<BE> + GetAutomorphismGaloisElement + GGLWEInfos,
    M: ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let log_n: usize = module.log_n();

    if i == log_n {
        return;
    }

    // Isolate the first accumulator
    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    // Control = true accumlator is free to overide
    if !acc_prev[0].control {
        let acc_mut_ref: &mut Accumulator = &mut acc_prev[0]; // from split_at_mut

        // No previous value -> copies and sets flags accordingly
        if let Some(a_ref) = a {
            module.glwe_copy(&mut acc_mut_ref.data, a_ref);
            acc_mut_ref.value = true
        } else {
            acc_mut_ref.value = false
        }
        acc_mut_ref.control = true; // Able to be combined on next call
    } else {
        // Compresses acc_prev <- combine(acc_prev, a).
        combine(module, &mut acc_prev[0], a, i, auto_keys, scratch);
        acc_prev[0].control = false;

        // Propagates to next accumulator
        if acc_prev[0].value {
            pack_core(
                module,
                Some(&acc_prev[0].data),
                acc_next,
                i + 1,
                auto_keys,
                scratch,
            );
        } else {
            pack_core(
                module,
                None::<&GLWE<Vec<u8>>>,
                acc_next,
                i + 1,
                auto_keys,
                scratch,
            );
        }
    }
}

/// [combine] merges two ciphertexts together.
fn combine<B, M, K, BE: Backend>(
    module: &M,
    acc: &mut Accumulator,
    b: Option<&B>,
    i: usize,
    auto_keys: &HashMap<i64, K>,
    scratch: &mut Scratch<BE>,
) where
    B: GLWEToRef + GLWEInfos,
    M: GLWEAutomorphism<BE> + GaloisElement + GLWERotate<BE> + GLWESub + GLWEShift<BE> + GLWEAdd + GLWENormalize<BE>,
    B: GLWEToRef + GLWEInfos,
    K: GGLWEPreparedToRef<BE> + GetAutomorphismGaloisElement + GGLWEInfos,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let log_n: usize = acc.data.n().log2();
    let a: &mut GLWE<Vec<u8>> = &mut acc.data;

    let gal_el: i64 = if i == 0 {
        -1
    } else {
        module.galois_element(1 << (i - 1))
    };

    let t: i64 = 1 << (log_n - i - 1);

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
    if acc.value {
        if let Some(b) = b {
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(module, a);

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
            if let Some(auto_key) = auto_keys.get(&gal_el) {
                module.glwe_automorphism_inplace(&mut tmp_b, auto_key, scratch_1);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }

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
            if let Some(auto_key) = auto_keys.get(&gal_el) {
                module.glwe_automorphism_add_inplace(a, auto_key, scratch);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }
        }
    } else if let Some(b) = b {
        let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(module, a);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        if let Some(auto_key) = auto_keys.get(&gal_el) {
            module.glwe_automorphism_sub_negate(a, &tmp_b, auto_key, scratch_1);
        } else {
            panic!("auto_key[{gal_el}] not found");
        }

        acc.value = true;
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
    K: GGLWEPreparedToRef<BE> + GetAutomorphismGaloisElement + GGLWEInfos,
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
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(module, a);

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

        let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(module, b);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        module.glwe_automorphism_sub_negate(b, &tmp_b, auto_key, scratch_1);
    }
}
