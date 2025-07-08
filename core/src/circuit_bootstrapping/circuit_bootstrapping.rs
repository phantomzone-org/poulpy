use std::{collections::HashMap, time::Instant, usize};

use crate::{
    BlindRotationKeyCGGI, FourierGLWESecret, GGSWCiphertext, GLWEAutomorphismKey, GLWECiphertext, GLWEOps, GLWESecret,
    GLWETensorKey, Infos, LWECiphertext, LWESecret, LookUpTable, LookUpTableRotationDirection, ScratchCore, SetRow,
    cggi_blind_rotate,
};
use backend::{FFT64, Module, Scratch};
use sampling::source::Source;

pub struct CircuitBootstrappingKeyCGGI<D> {
    pub(crate) brk: BlindRotationKeyCGGI<D, FFT64>,
    pub(crate) tsk: GLWETensorKey<Vec<u8>, FFT64>,
    pub(crate) auto_keys: HashMap<i64, GLWEAutomorphismKey<Vec<u8>, FFT64>>,
}

impl CircuitBootstrappingKeyCGGI<Vec<u8>> {
    pub fn generate<DLwe, DGlwe>(
        module: &Module<FFT64>,
        basek: usize,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        k_brk: usize,
        rows_brk: usize,
        k_trace: usize,
        rows_trace: usize,
        k_tsk: usize,
        rows_tsk: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) -> Self
    where
        DLwe: AsRef<[u8]>,
        DGlwe: AsRef<[u8]>,
    {
        let mut auto_keys: HashMap<i64, GLWEAutomorphismKey<Vec<u8>, FFT64>> = HashMap::new();
        let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(&module);
        gal_els.iter().for_each(|gal_el| {
            let mut key: GLWEAutomorphismKey<Vec<u8>, FFT64> =
                GLWEAutomorphismKey::alloc(&module, basek, k_trace, rows_trace, 1, sk_glwe.rank());
            key.encrypt_sk(
                &module, *gal_el, &sk_glwe, source_xa, source_xe, sigma, scratch,
            );
            auto_keys.insert(*gal_el, key);
        });

        let mut sk_glwe_fourier: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::alloc(module, sk_glwe.rank());
        sk_glwe_fourier.set(module, &sk_glwe);

        let mut brk: BlindRotationKeyCGGI<Vec<u8>, FFT64> =
            BlindRotationKeyCGGI::allocate(module, sk_lwe.n(), basek, k_brk, rows_brk, sk_glwe.rank());
        brk.generate_from_sk(
            module,
            &sk_glwe_fourier,
            sk_lwe,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );

        let mut tsk: GLWETensorKey<Vec<u8>, FFT64> = GLWETensorKey::alloc(module, basek, k_tsk, rows_tsk, 1, sk_glwe.rank());
        tsk.encrypt_sk(
            module,
            &sk_glwe_fourier,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );

        Self {
            brk,
            auto_keys,
            tsk,
        }
    }
}

pub fn circuit_bootstrap_to_constant_cggi<DRes, DLwe, DBrk>(
    module: &Module<FFT64>,
    res: &mut GGSWCiphertext<DRes, FFT64>,
    lwe: &LWECiphertext<DLwe>,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyCGGI<DBrk>,
    scratch: &mut Scratch,
) where
    DRes: AsRef<[u8]> + AsMut<[u8]>,
    DLwe: AsRef<[u8]>,
    DBrk: AsRef<[u8]>,
{
    circuit_bootstrap_core_cggi(
        false,
        module,
        0,
        res,
        lwe,
        log_domain,
        extension_factor,
        key,
        scratch,
    );
}

pub fn circuit_bootstrap_to_exponent_cggi<DRes, DLwe, DBrk>(
    module: &Module<FFT64>,
    log_gap_out: usize,
    res: &mut GGSWCiphertext<DRes, FFT64>,
    lwe: &LWECiphertext<DLwe>,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyCGGI<DBrk>,
    scratch: &mut Scratch,
) where
    DRes: AsRef<[u8]> + AsMut<[u8]>,
    DLwe: AsRef<[u8]>,
    DBrk: AsRef<[u8]>,
{
    circuit_bootstrap_core_cggi(
        true,
        module,
        log_gap_out,
        res,
        lwe,
        log_domain,
        extension_factor,
        key,
        scratch,
    );
}

pub fn circuit_bootstrap_core_cggi<DRes, DLwe, DBrk>(
    to_exponent: bool,
    module: &Module<FFT64>,
    log_gap_out: usize,
    res: &mut GGSWCiphertext<DRes, FFT64>,
    lwe: &LWECiphertext<DLwe>,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyCGGI<DBrk>,
    scratch: &mut Scratch,
) where
    DRes: AsRef<[u8]> + AsMut<[u8]>,
    DLwe: AsRef<[u8]>,
    DBrk: AsRef<[u8]>,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), module.n());
        assert_eq!(key.brk.n(), module.n());
        assert_eq!(lwe.basek(), key.brk.basek());
        assert_eq!(res.basek(), key.brk.basek());
    }

    let basek: usize = res.basek();
    let rows: usize = res.rows();
    let rank: usize = res.rank();
    let k: usize = res.k();

    let alpha: usize = rows.next_power_of_two();

    let mut f: Vec<i64> = vec![0i64; (1 << log_domain) * alpha];

    if to_exponent {
        (0..rows).for_each(|i| {
            f[i] = 1 << (basek * (rows - 1 - i));
        });
    } else {
        (0..1 << log_domain).for_each(|j| {
            (0..rows).for_each(|i| {
                f[j * alpha + i] = j as i64 * (1 << (basek * (rows - 1 - i)));
            });
        });
    }

    // Lut precision, basically must be able to hold the decomposition power basis of the GGSW
    let mut lut: LookUpTable = LookUpTable::alloc(module, basek, k, extension_factor);
    lut.set(module, &f, basek * rows);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    // TODO: separate GGSW k from output of blind rotation k
    let (mut res_glwe, scratch1) = scratch.tmp_glwe_ct(module, basek, k, rank);
    let (mut tmp_res, scratch2) = scratch1.tmp_glwe_ct(module, basek, k, rank);
    let (mut tmp_dft, scratch3) = scratch2.tmp_fourier_glwe_ct(module, basek, k, rank);
    let (mut tmp_gglwe, scratch4) = scratch3.tmp_gglwe(module, basek, k, rows, 1, rank, rank);

    let now: Instant = Instant::now();
    cggi_blind_rotate(module, &mut res_glwe, &lwe, &lut, &key.brk, scratch4);
    println!("cggi_blind_rotate: {} ms", now.elapsed().as_millis());

    let gap: usize = 2 * lut.drift / lut.extension_factor();

    let log_gap_in: usize = (usize::BITS - (gap * alpha - 1).leading_zeros()) as _;

    (0..rows).for_each(|i| {
        if to_exponent {
            let now: Instant = Instant::now();
            // Isolates i-th LUT and moves coefficients according to requested gap.
            post_process(
                module,
                &mut tmp_res,
                &res_glwe,
                log_gap_in,
                log_gap_out,
                log_domain,
                &key.auto_keys,
                scratch4,
            );
            println!("post_process: {} ms", now.elapsed().as_millis());
        } else {
            tmp_res.trace(
                module,
                0,
                module.log_n(),
                &res_glwe,
                &key.auto_keys,
                scratch4,
            );
        }

        // Switches
        tmp_res.dft(module, &mut tmp_dft);
        tmp_gglwe.set_row(module, i, 0, &tmp_dft);

        if i < rows {
            res_glwe.rotate_inplace(module, -(gap as i64));
        }
    });

    // Expands GGLWE to GGSW using GGLWE(s^2)
    res.from_gglwe(module, &tmp_gglwe, &key.tsk, scratch4);
}

fn post_process<DataRes, DataA>(
    module: &Module<FFT64>,
    res: &mut GLWECiphertext<DataRes>,
    a: &GLWECiphertext<DataA>,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &HashMap<i64, GLWEAutomorphismKey<Vec<u8>, FFT64>>,
    scratch: &mut Scratch,
) where
    DataRes: AsMut<[u8]> + AsRef<[u8]>,
    DataA: AsRef<[u8]>,
{
    let log_n: usize = module.log_n();

    let mut cts: HashMap<usize, GLWECiphertext<Vec<u8>>> = HashMap::new();

    // First partial trace, vanishes all coefficients which are not multiples of gap_in
    // [1, 1, 1, 1, 0, 0, 0, ..., 0, 0, -1, -1, -1, -1] -> [1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0]
    res.trace(
        module,
        module.log_n() - log_gap_in as usize + 1,
        log_n,
        &a,
        auto_keys,
        scratch,
    );

    // TODO: optimize with packing and final partial trace
    // If gap_out < gap_in, then we need to repack, i.e. reduce the cap between coefficients.
    if log_gap_in != log_gap_out {
        let steps: i32 = 1 << log_domain;
        (0..steps).for_each(|i| {
            if i != 0 {
                res.rotate_inplace(module, -(1 << log_gap_in));
            }
            cts.insert(i as usize * (1 << log_gap_out), res.clone());
        });

        let now: Instant = Instant::now();
        pack(module, &mut cts, log_gap_out, auto_keys, scratch);
        println!("pack: {} ms", now.elapsed().as_millis());
        let packed: GLWECiphertext<Vec<u8>> = cts.remove(&0).unwrap();
        res.trace(
            module,
            log_n - log_gap_out,
            log_n,
            &packed,
            auto_keys,
            scratch,
        );
    }
}

pub fn pack<D>(
    module: &Module<FFT64>,
    cts: &mut HashMap<usize, GLWECiphertext<D>>,
    log_gap_out: usize,
    auto_keys: &HashMap<i64, GLWEAutomorphismKey<Vec<u8>, FFT64>>,
    scratch: &mut Scratch,
) where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    let log_n: usize = module.log_n();

    let basek: usize = cts.get(&0).unwrap().basek();
    let k: usize = cts.get(&0).unwrap().k();
    let rank: usize = cts.get(&0).unwrap().rank();

    (0..log_n - log_gap_out).for_each(|i| {
        let now: Instant = Instant::now();

        let t = 16.min(1 << (log_n - 1 - i));

        let auto_key: &GLWEAutomorphismKey<Vec<u8>, FFT64>;
        if i == 0 {
            auto_key = auto_keys.get(&-1).unwrap()
        } else {
            auto_key = auto_keys.get(&module.galois_element(1 << (i - 1))).unwrap();
        }

        (0..t).for_each(|j| {
            let mut a: Option<GLWECiphertext<D>> = cts.remove(&j);
            let mut b: Option<GLWECiphertext<D>> = cts.remove(&(j + t));

            combine(
                module,
                basek,
                k,
                rank,
                a.as_mut(),
                b.as_mut(),
                i,
                auto_key,
                scratch,
            );

            if let Some(a) = a {
                cts.insert(j, a);
            } else if let Some(b) = b {
                cts.insert(j, b);
            }
        });

        println!("combine: {} us", now.elapsed().as_micros());
    });
}

fn combine<A: AsMut<[u8]> + AsRef<[u8]>, B: AsRef<[u8]> + AsMut<[u8]>, DataAK: AsRef<[u8]>>(
    module: &Module<FFT64>,
    basek: usize,
    k: usize,
    rank: usize,
    a: Option<&mut GLWECiphertext<A>>,
    b: Option<&mut GLWECiphertext<B>>,
    i: usize,
    auto_key: &GLWEAutomorphismKey<DataAK, FFT64>,
    scratch: &mut Scratch,
) {
    let log_n: usize = module.log_n();
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
    if let Some(a) = a {
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
            tmp_b.automorphism_inplace(module, auto_key, scratch_1);

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
            a.automorphism_add_inplace(module, auto_key, scratch);
        }
    } else {
        if let Some(b) = b {
            let (mut tmp_b, scratch_1) = scratch.tmp_glwe_ct(module, basek, k, rank);
            tmp_b.rotate(module, 1 << (log_n - i - 1), b);
            tmp_b.rsh(1, scratch_1);

            // a = (b* X^t - phi(b* X^t))
            b.automorphism_sub_ba(module, &tmp_b, auto_key, scratch_1);
        }
    }
}
