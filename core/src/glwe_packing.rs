use crate::{AutomorphismKey, GLWECiphertext, GLWEOps, Infos, ScratchCore};
use std::collections::HashMap;

use backend::{FFT64, Module, Scratch};

/// [StreamPacker] enables only the fly GLWE packing
/// with constant memory of Log(N) ciphertexts.
/// Main difference with usual GLWE packing is that
/// the output is bit-reversed.
pub struct StreamPacker {
    accumulators: Vec<Accumulator>,
    log_batch: usize,
    counter: usize,
}

/// [Accumulator] stores intermediate packing result.
/// There are Log(N) such accumulators in a [StreamPacker].
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
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: GLWECiphertext::alloc(module, basek, k, rank),
            value: false,
            control: false,
        }
    }
}

impl StreamPacker {
    /// Instantiates a new [StreamPacker].
    ///
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `log_batch`: packs coefficients which are multiples of X^{N/2^log_batch}.
    ///                i.e. with `log_batch=0` only the constant coefficient is packed
    ///                and N GLWE ciphertext can be packed. With `log_batch=2` all coefficients
    ///                which are multiples of X^{N/4} are packed. Meaning that N/4 ciphertexts
    ///                can be packed.
    /// * `basek`: base 2 logarithm of the GLWE ciphertext in memory digit representation.
    /// * `k`: base 2 precision of the GLWE ciphertext precision over the Torus.
    /// * `rank`: rank of the GLWE ciphertext.
    pub fn new(module: &Module<FFT64>, log_batch: usize, basek: usize, k: usize, rank: usize) -> Self {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = module.log_n();
        (0..log_n - log_batch).for_each(|_| accumulators.push(Accumulator::alloc(module, basek, k, rank)));
        Self {
            accumulators: accumulators,
            log_batch,
            counter: 0,
        }
    }

    /// Implicit reset of the internal state (to be called before a new packing procedure).
    pub fn reset(&mut self) {
        for i in 0..self.accumulators.len() {
            self.accumulators[i].value = false;
            self.accumulators[i].control = false;
        }
        self.counter = 0;
    }

    /// Number of scratch space bytes required to call [Self::add].
    pub fn scratch_space(module: &Module<FFT64>, basek: usize, ct_k: usize, atk_k: usize, rank: usize) -> usize {
        pack_core_scratch_space(module, basek, ct_k, atk_k, rank)
    }

    pub fn galois_elements(module: &Module<FFT64>) -> Vec<i64> {
        GLWECiphertext::trace_galois_elements(module)
    }

    /// Adds a GLWE ciphertext to the [StreamPacker]. And propagates
    /// intermediate results among the [Accumulator]s.
    ///
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `res`: space to append fully packed ciphertext. Only when the number
    ///          of packed ciphertexts reaches N/2^log_batch is a result written.
    /// * `a`: ciphertext to pack. Can optionally give None to pack a 0 ciphertext.
    /// * `auto_keys`: a [HashMap] containing the [AutomorphismKey]s.
    /// * `scratch`: scratch space of size at least [Self::add_scratch_space].
    pub fn add<DataA: AsRef<[u8]>, DataAK: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        res: &mut Vec<GLWECiphertext<Vec<u8>>>,
        a: Option<&GLWECiphertext<DataA>>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) {
        pack_core(
            module,
            a,
            &mut self.accumulators,
            self.log_batch,
            auto_keys,
            scratch,
        );
        self.counter += 1 << self.log_batch;
        if self.counter == module.n() {
            res.push(
                self.accumulators[module.log_n() - self.log_batch - 1]
                    .data
                    .clone(),
            );
            self.reset();
        }
    }

    /// Flushes all accumlators and appends the result to `res`.
    pub fn flush<DataAK: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        res: &mut Vec<GLWECiphertext<Vec<u8>>>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) {
        if self.counter != 0 {
            while self.counter != 0 {
                self.add(
                    module,
                    res,
                    None::<&GLWECiphertext<Vec<u8>>>,
                    auto_keys,
                    scratch,
                );
            }
        }
    }
}

fn pack_core_scratch_space(module: &Module<FFT64>, basek: usize, ct_k: usize, atk_k: usize, rank: usize) -> usize {
    combine_scratch_space(module, basek, ct_k, atk_k, rank)
}

fn pack_core<D: AsRef<[u8]>, DataAK: AsRef<[u8]>>(
    module: &Module<FFT64>,
    a: Option<&GLWECiphertext<D>>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
    scratch: &mut Scratch,
) {
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

fn combine_scratch_space(module: &Module<FFT64>, basek: usize, ct_k: usize, atk_k: usize, rank: usize) -> usize {
    GLWECiphertext::bytes_of(module, basek, ct_k, rank)
        + (GLWECiphertext::rsh_scratch_space(module)
            | GLWECiphertext::automorphism_scratch_space(module, basek, ct_k, ct_k, atk_k, rank))
}

/// [combine] merges two ciphertexts together.
fn combine<D: AsRef<[u8]>, DataAK: AsRef<[u8]>>(
    module: &Module<FFT64>,
    acc: &mut Accumulator,
    b: Option<&GLWECiphertext<D>>,
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
    scratch: &mut Scratch,
) {
    let log_n: usize = module.log_n();
    let a: &mut GLWECiphertext<Vec<u8>> = &mut acc.data;
    let basek: usize = a.basek();
    let k: usize = a.k();
    let rank: usize = a.rank();

    let gal_el: i64;

    if i == 0 {
        gal_el = -1;
    } else {
        gal_el = module.galois_element(1 << (i - 1))
    }

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
            let (mut tmp_b, scratch_1) = scratch.tmp_glwe_ct(module, basek, k, rank);

            // a = a * X^-t
            a.rotate_inplace(module, -t);

            // tmp_b = a * X^-t - b
            tmp_b.sub(module, a, b);
            tmp_b.rsh(1, scratch_1);

            // a = a * X^-t + b
            a.add_inplace(module, b);
            a.rsh(1, scratch_1);

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
            a.rsh(1, scratch);
            // a = a + phi(a)
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }
        }
    } else {
        if let Some(b) = b {
            let (mut tmp_b, scratch_1) = scratch.tmp_glwe_ct(module, basek, k, rank);
            tmp_b.rotate(module, 1 << (log_n - i - 1), b);
            tmp_b.rsh(1, scratch_1);

            // a = (b* X^t - phi(b* X^t))
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_sub_ba::<&mut [u8], _>(module, &tmp_b, key, scratch_1);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }

            acc.value = true;
        }
    }
}
