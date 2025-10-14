use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNegateInplace,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace,
        VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    GLWEOperations, TakeGLWE,
    layouts::{GGLWEInfos, GLWE, GLWEInfos, LWEInfos, prepared::AutomorphismKeyPrepared},
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
    /// * `module`: static backend FFT tables.
    /// * `log_batch`: packs coefficients which are multiples of X^{N/2^log_batch}.
    ///   i.e. with `log_batch=0` only the constant coefficient is packed
    ///   and N GLWE ciphertext can be packed. With `log_batch=2` all coefficients
    ///   which are multiples of X^{N/4} are packed. Meaning that N/4 ciphertexts
    ///   can be packed.
    pub fn new<A>(infos: &A, log_batch: usize) -> Self
    where
        A: GLWEInfos,
    {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = infos.n().log2();
        (0..log_n - log_batch).for_each(|_| accumulators.push(Accumulator::alloc(infos)));
        Self {
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
    pub fn scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
    where
        OUT: GLWEInfos,
        KEY: GGLWEInfos,
        Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        pack_core_scratch_space(module, out_infos, key_infos)
    }

    pub fn galois_elements<B: Backend>(module: &Module<B>) -> Vec<i64> {
        GLWE::trace_galois_elements(module)
    }

    /// Adds a GLWE ciphertext to the [GLWEPacker].
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `res`: space to append fully packed ciphertext. Only when the number
    ///   of packed ciphertexts reaches N/2^log_batch is a result written.
    /// * `a`: ciphertext to pack. Can optionally give None to pack a 0 ciphertext.
    /// * `auto_keys`: a [HashMap] containing the [AutomorphismKeyExec]s.
    /// * `scratch`: scratch space of size at least [Self::scratch_space].
    pub fn add<DataA: DataRef, DataAK: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: Option<&GLWE<DataA>>,
        auto_keys: &HashMap<i64, AutomorphismKeyPrepared<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftBytesOf
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace<B>
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace<B>
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
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
    pub fn flush<Data: DataMut, B: Backend>(&mut self, module: &Module<B>, res: &mut GLWE<Data>)
    where
        Module<B>: VecZnxCopy,
    {
        assert!(self.counter as u32 == self.accumulators[0].data.n());
        // Copy result GLWE into res GLWE
        res.copy(
            module,
            &self.accumulators[module.log_n() - self.log_batch - 1].data,
        );

        self.reset();
    }
}

fn pack_core_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
where
    OUT: GLWEInfos,
    KEY: GGLWEInfos,
    Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
{
    combine_scratch_space(module, out_infos, key_infos)
}

fn pack_core<D: DataRef, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    a: Option<&GLWE<D>>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKeyPrepared<DataAK, B>>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxCopy
        + VecZnxRotateInplace<B>
        + VecZnxSub
        + VecZnxNegateInplace
        + VecZnxRshInplace<B>
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxSubInplace
        + VecZnxRotate
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeTmpBytes,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
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
            acc_mut_ref.data.copy(module, a_ref);
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

fn combine_scratch_space<B: Backend, OUT, KEY>(module: &Module<B>, out_infos: &OUT, key_infos: &KEY) -> usize
where
    OUT: GLWEInfos,
    KEY: GGLWEInfos,
    Module<B>: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
{
    GLWE::bytes_of(out_infos)
        + (GLWE::rsh_scratch_space(module.n()) | GLWE::automorphism_inplace_scratch_space(module, out_infos, key_infos))
}

/// [combine] merges two ciphertexts together.
fn combine<D: DataRef, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    acc: &mut Accumulator,
    b: Option<&GLWE<D>>,
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKeyPrepared<DataAK, B>>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxCopy
        + VecZnxRotateInplace<B>
        + VecZnxSub
        + VecZnxNegateInplace
        + VecZnxRshInplace<B>
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxSubInplace
        + VecZnxRotate
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxNormalize<B>
        + VecZnxNormalizeTmpBytes,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeGLWE,
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
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(a);

            // a = a * X^-t
            a.rotate_inplace(module, -t, scratch_1);

            // tmp_b = a * X^-t - b
            tmp_b.sub(module, a, b);
            tmp_b.rsh(module, 1, scratch_1);

            // a = a * X^-t + b
            a.add_inplace(module, b);
            a.rsh(module, 1, scratch_1);

            tmp_b.normalize_inplace(module, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            if let Some(key) = auto_keys.get(&gal_el) {
                tmp_b.automorphism_inplace(module, key, scratch_1);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }

            // a = a * X^-t + b - phi(a * X^-t - b)
            a.sub_inplace_ab(module, &tmp_b);
            a.normalize_inplace(module, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            a.rotate_inplace(module, t, scratch_1);
        } else {
            a.rsh(module, 1, scratch);
            // a = a + phi(a)
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }
        }
    } else if let Some(b) = b {
        let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(a);
        tmp_b.rotate(module, 1 << (log_n - i - 1), b);
        tmp_b.rsh(module, 1, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        if let Some(key) = auto_keys.get(&gal_el) {
            a.automorphism_sub_negate(module, &tmp_b, key, scratch_1);
        } else {
            panic!("auto_key[{gal_el}] not found");
        }

        acc.value = true;
    }
}

/// Packs [x_0: GLWE(m_0), x_1: GLWE(m_1), ..., x_i: GLWE(m_i)]
/// to [0: GLWE(m_0 * X^x_0 + m_1 * X^x_1 + ... + m_i * X^x_i)]
pub fn glwe_packing<D: DataMut, ATK, B: Backend>(
    module: &Module<B>,
    cts: &mut HashMap<usize, &mut GLWE<D>>,
    log_gap_out: usize,
    auto_keys: &HashMap<i64, AutomorphismKeyPrepared<ATK, B>>,
    scratch: &mut Scratch<B>,
) where
    ATK: DataRef,
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubInplace
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxRotate
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnx + TakeVecZnxDft<B> + ScratchAvailable,
{
    #[cfg(debug_assertions)]
    {
        assert!(*cts.keys().max().unwrap() < module.n())
    }

    let log_n: usize = module.log_n();

    (0..log_n - log_gap_out).for_each(|i| {
        let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

        let auto_key: &AutomorphismKeyPrepared<ATK, B> = if i == 0 {
            auto_keys.get(&-1).unwrap()
        } else {
            auto_keys.get(&module.galois_element(1 << (i - 1))).unwrap()
        };

        (0..t).for_each(|j| {
            let mut a: Option<&mut GLWE<D>> = cts.remove(&j);
            let mut b: Option<&mut GLWE<D>> = cts.remove(&(j + t));

            pack_internal(module, &mut a, &mut b, i, auto_key, scratch);

            if let Some(a) = a {
                cts.insert(j, a);
            } else if let Some(b) = b {
                cts.insert(j, b);
            }
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn pack_internal<A: DataMut, D: DataMut, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    a: &mut Option<&mut GLWE<A>>,
    b: &mut Option<&mut GLWE<D>>,
    i: usize,
    auto_key: &AutomorphismKeyPrepared<DataAK, B>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubInplace
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxRotate
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnx + TakeVecZnxDft<B> + ScratchAvailable,
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
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(a);

            // a = a * X^-t
            a.rotate_inplace(module, -t, scratch_1);

            // tmp_b = a * X^-t - b
            tmp_b.sub(module, a, b);
            tmp_b.rsh(module, 1, scratch_1);

            // a = a * X^-t + b
            a.add_inplace(module, b);
            a.rsh(module, 1, scratch_1);

            tmp_b.normalize_inplace(module, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            tmp_b.automorphism_inplace(module, auto_key, scratch_1);

            // a = a * X^-t + b - phi(a * X^-t - b)
            a.sub_inplace_ab(module, &tmp_b);
            a.normalize_inplace(module, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            a.rotate_inplace(module, t, scratch_1);
        } else {
            a.rsh(module, 1, scratch);
            // a = a + phi(a)
            a.automorphism_add_inplace(module, auto_key, scratch);
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(b);
        tmp_b.rotate(module, t, b);
        tmp_b.rsh(module, 1, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        b.automorphism_sub_negate(module, &tmp_b, auto_key, scratch_1);
    }
}
