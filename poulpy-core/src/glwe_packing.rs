use std::collections::HashMap;

use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAutomorphismInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAllocBytes, VecZnxNegateInplace, VecZnxNormalizeInplace, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    GLWEOperations, TakeGLWECt,
    layouts::{GLWECiphertext, Infos, prepared::GGLWEAutomorphismKeyPrepared},
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
    data: GLWECiphertext<Vec<u8>>,
    value: bool,   // Implicit flag for zero ciphertext
    control: bool, // Can be combined with incoming value
}

impl Accumulator {
    /// Allocates a new [Accumulator].
    ///
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `basek`: base 2 logarithm of the GLWE ciphertext in memory digit representation.
    /// * `k`: base 2 precision of the GLWE ciphertext precision over the Torus.
    /// * `rank`: rank of the GLWE ciphertext.
    pub fn alloc(n: usize, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: GLWECiphertext::alloc(n, basek, k, rank),
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
    /// * `basek`: base 2 logarithm of the GLWE ciphertext in memory digit representation.
    /// * `k`: base 2 precision of the GLWE ciphertext precision over the Torus.
    /// * `rank`: rank of the GLWE ciphertext.
    pub fn new(n: usize, log_batch: usize, basek: usize, k: usize, rank: usize) -> Self {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = (usize::BITS - (n - 1).leading_zeros()) as _;
        (0..log_n - log_batch).for_each(|_| accumulators.push(Accumulator::alloc(n, basek, k, rank)));
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
    pub fn scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        ct_k: usize,
        k_ksk: usize,
        digits: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    {
        pack_core_scratch_space(module, basek, ct_k, k_ksk, digits, rank)
    }

    pub fn galois_elements<B: Backend>(module: &Module<B>) -> Vec<i64> {
        GLWECiphertext::trace_galois_elements(module)
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
        a: Option<&GLWECiphertext<DataA>>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<DataAK, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApply<B>
            + VmpApplyAdd<B>
            + DFT<B>
            + IDFTConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubABInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxBigAutomorphismInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        assert!(
            self.counter < self.accumulators[0].data.n(),
            "Packing limit of {} reached",
            self.accumulators[0].data.n() >> self.log_batch
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
    pub fn flush<Data: DataMut, B: Backend>(&mut self, module: &Module<B>, res: &mut GLWECiphertext<Data>)
    where
        Module<B>: VecZnxCopy,
    {
        assert!(self.counter == self.accumulators[0].data.n());
        // Copy result GLWE into res GLWE
        res.copy(
            module,
            &self.accumulators[module.log_n() - self.log_batch - 1].data,
        );

        self.reset();
    }
}

fn pack_core_scratch_space<B: Backend>(
    module: &Module<B>,
    basek: usize,
    ct_k: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
) -> usize
where
    Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
{
    combine_scratch_space(module, basek, ct_k, k_ksk, digits, rank)
}

fn pack_core<D: DataRef, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    a: Option<&GLWECiphertext<D>>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<DataAK, B>>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + DFT<B>
        + IDFTConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxCopy
        + VecZnxRotateInplace
        + VecZnxSub
        + VecZnxNegateInplace
        + VecZnxRshInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxSubABInplace
        + VecZnxRotate
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxBigAutomorphismInplace<B>,
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
                None::<&GLWECiphertext<Vec<u8>>>,
                acc_next,
                i + 1,
                auto_keys,
                scratch,
            );
        }
    }
}

fn combine_scratch_space<B: Backend>(
    module: &Module<B>,
    basek: usize,
    ct_k: usize,
    k_ksk: usize,
    digits: usize,
    rank: usize,
) -> usize
where
    Module<B>: VecZnxDftAllocBytes + VmpApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
{
    GLWECiphertext::bytes_of(module.n(), basek, ct_k, rank)
        + (GLWECiphertext::rsh_scratch_space(module.n())
            | GLWECiphertext::automorphism_scratch_space(module, basek, ct_k, ct_k, k_ksk, digits, rank))
}

/// [combine] merges two ciphertexts together.
fn combine<D: DataRef, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    acc: &mut Accumulator,
    b: Option<&GLWECiphertext<D>>,
    i: usize,
    auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<DataAK, B>>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + DFT<B>
        + IDFTConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxCopy
        + VecZnxRotateInplace
        + VecZnxSub
        + VecZnxNegateInplace
        + VecZnxRshInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxSubABInplace
        + VecZnxRotate
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxBigAutomorphismInplace<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    let n: usize = acc.data.n();
    let log_n: usize = (u64::BITS - (n - 1).leading_zeros()) as _;
    let a: &mut GLWECiphertext<Vec<u8>> = &mut acc.data;
    let basek: usize = a.basek();
    let k: usize = a.k();
    let rank: usize = a.rank();

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
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(n, basek, k, rank);

            // a = a * X^-t
            a.rotate_inplace(module, -t);

            // tmp_b = a * X^-t - b
            tmp_b.sub(module, a, b);
            tmp_b.rsh(module, 1);

            // a = a * X^-t + b
            a.add_inplace(module, b);
            a.rsh(module, 1);

            tmp_b.normalize_inplace(module, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            if let Some(key) = auto_keys.get(&gal_el) {
                tmp_b.automorphism_inplace(module, key, scratch_1);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }

            // a = a * X^-t + b - phi(a * X^-t - b)
            a.sub_inplace_ab(module, &tmp_b);
            a.normalize_inplace(module, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            a.rotate_inplace(module, t);
        } else {
            a.rsh(module, 1);
            // a = a + phi(a)
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }
        }
    } else if let Some(b) = b {
        let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(n, basek, k, rank);
        tmp_b.rotate(module, 1 << (log_n - i - 1), b);
        tmp_b.rsh(module, 1);

        // a = (b* X^t - phi(b* X^t))
        if let Some(key) = auto_keys.get(&gal_el) {
            a.automorphism_sub_ba(module, &tmp_b, key, scratch_1);
        } else {
            panic!("auto_key[{}] not found", gal_el);
        }

        acc.value = true;
    }
}
