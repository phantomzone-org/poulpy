//! End-to-end CKKS example for evaluating a cubic polynomial
//!
//! This example evaluates
//!
//! `p(x) = (a + b*x) + (c + d*x) * x^2`
//!
//! over an encrypted complex slot vector. The example is intentionally split
//! into six explicit phases:
//!
//! 1. `setup`
//! 2. `encoding`
//! 3. `encryption`
//! 4. `evaluation`
//! 5. `decryption`
//! 6. `verification`
//!
//! Each phase is implemented as its own function so the control flow matches
//! the usual CKKS workflow and the metadata transitions are easy to inspect.

use anyhow::Result;
use poulpy_ckks::{
    CKKSInfos, CKKSMeta,
    encoding::Encoder,
    layouts::{CKKSCiphertext, CKKSMaintainOps, CKKSModuleAlloc, CKKSPlaintext},
    leveled::api::{CKKSAddOpsUnsafe, CKKSAllOpsTmpBytes, CKKSDecrypt, CKKSEncrypt, CKKSMulAddOps, CKKSMulOps},
};
use poulpy_core::{
    EncryptionLayout, GLWENormalize, GLWETensorKeyEncryptSk,
    layouts::{
        GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, GLWEToBackendMut, LWEInfos, ModuleCoreAlloc, Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_ref::NTT120Ref;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

type BakcendImpl = NTT120Ref;
type SecretKeyPrepared = GLWESecretPrepared<<BakcendImpl as Backend>::OwnedBuf, BakcendImpl>;
type TensorKeyPrepared = GLWETensorKeyPrepared<<BakcendImpl as Backend>::OwnedBuf, BakcendImpl>;

const N: usize = 4096;
const M: usize = N / 2;
const BASE2K: usize = 52;
const CT_K: usize = 95;
const HW: usize = 192;
const DSIZE: usize = 1;
const PREC_CT: CKKSMeta = CKKSMeta {
    log_delta: 30,
    log_budget: 5,
};
const PREC_PT: CKKSMeta = CKKSMeta {
    log_delta: 4,
    log_budget: 0,
};

/// Long-lived objects prepared during setup and reused by the later phases.
struct SetupArtifacts {
    module: Module<BakcendImpl>,
    encoder: Encoder<f64>,
    sk: SecretKeyPrepared,
    tsk_prepared: TensorKeyPrepared,
    scratch: ScratchOwned<BakcendImpl>,
}

/// Complex coefficients of the polynomial `(a + b*x) + (c + d*x) * x^2`.
struct PolynomialCoeffs {
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    d: (f64, f64),
}

/// Inputs prepared by the encoding phase.
///
/// This includes both the encoded slot vector `x` and the plaintext constants
/// used later during evaluation.
struct EncodingArtifacts {
    x_re: Vec<f64>,
    x_im: Vec<f64>,
    coeffs: PolynomialCoeffs,
    cst_a: CKKSPlaintext<Vec<u8>>,
    cst_b: CKKSPlaintext<Vec<u8>>,
    cst_c: CKKSPlaintext<Vec<u8>>,
    cst_d: CKKSPlaintext<Vec<u8>>,
    pt_znx: CKKSPlaintext<Vec<u8>>,
}

/// Ciphertexts produced by the encryption phase.
struct EncryptionArtifacts {
    ct_x: CKKSCiphertext<Vec<u8>>,
}

/// Final ciphertext produced by homomorphic evaluation.
struct EvaluationArtifacts {
    poly: CKKSCiphertext<Vec<u8>>,
}

/// Decoded slot values recovered after decryption.
struct DecryptionArtifacts {
    have_re: Vec<f64>,
    have_im: Vec<f64>,
}

fn encode_const_reim(
    module: &Module<BakcendImpl>,
    re: Option<f64>,
    im: Option<f64>,
    base2k: usize,
    prec: CKKSMeta,
) -> Result<CKKSPlaintext<Vec<u8>>> {
    let mut pt = module.ckks_pt_vec_znx_alloc(base2k.into(), prec);
    let n = module.n() as usize;
    let scale = 2f64.powi(prec.log_delta() as i32);
    let k = prec.effective_k().into();
    if prec.effective_k() <= 63 {
        let mut coeffs = vec![0i64; n];
        if let Some(re) = re {
            coeffs[0] = (re * scale).round() as i64;
        }
        if let Some(im) = im {
            coeffs[n / 2] = (im * scale).round() as i64;
        }
        pt.encode_vec_i64(&coeffs, k);
    } else {
        let mut coeffs = vec![0i128; n];
        if let Some(re) = re {
            coeffs[0] = (re * scale).round() as i128;
        }
        if let Some(im) = im {
            coeffs[n / 2] = (im * scale).round() as i128;
        }
        pt.encode_vec_i128(&coeffs, k);
    }
    Ok(pt)
}

/// Returns the GLWE ciphertext layout used throughout the example.
///
/// `CT_K` is the allocated ciphertext torus precision, while `BASE2K`
/// determines the limb size.
fn glwe_layout() -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    })
    .unwrap()
}

/// Returns the tensor-key layout used by ciphertext-ciphertext multiplication.
///
/// The tensor key is sized one decomposition block above the ciphertext
/// precision so multiplication and relinearization have enough headroom.
fn tsk_layout() -> EncryptionLayout<GLWETensorKeyLayout> {
    let k = CT_K + DSIZE * BASE2K;
    let dnum = CT_K.div_ceil(DSIZE * BASE2K);
    EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: k.into(),
        rank: Rank(1),
        dsize: DSIZE.into(),
        dnum: dnum.into(),
    })
    .unwrap()
}

/// Computes the maximum absolute error between two real vectors.
fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

/// Quantizes a complex coefficient onto the plaintext precision grid.
fn quantize_coeff(re: f64, im: f64) -> (f64, f64) {
    let scale = (PREC_PT.log_delta as f64).exp2();
    ((re * scale).round() / scale, (im * scale).round() / scale)
}

fn print_phase(name: &str) {
    println!("\n== {name} ==");
}

fn format_complex(re: f64, im: f64, digits: usize) -> String {
    format!("{re:.digits$}{im:+.digits$}i")
}

/// Prints the semantic and storage metadata of a CKKS ciphertext.
fn print_ct_meta(label: &str, ct: &CKKSCiphertext<Vec<u8>>) {
    println!(
        "  {label:<28} dec={:>2} hom={:>2} eff={:>3} limbs={:>2} max={:>3}",
        ct.log_delta(),
        ct.log_budget(),
        ct.effective_k(),
        ct.size(),
        ct.max_k().as_usize()
    );
}

/// Prints the semantic and storage metadata of a CKKS plaintext.
fn print_pt_meta(label: &str, pt: &CKKSPlaintext<Vec<u8>>) {
    println!(
        "  {label:<28} dec={:>2} hom={:>2} eff={:>3} limbs={:>2} max={:>3}",
        pt.log_delta(),
        pt.log_budget(),
        pt.effective_k(),
        pt.size(),
        pt.max_k().as_usize()
    );
}

/// Phase 1: setup.
///
/// This phase prepares all long-lived objects:
/// - the backend module
/// - the slot encoder
/// - the secret key
/// - the prepared tensor key used by multiplication
/// - one scratch buffer sized for the whole example
///
/// No message-dependent data is handled here yet.
fn setup() -> Result<SetupArtifacts> {
    print_phase("setup");
    println!("  polynomial: p(x) = (a + b*x) + (c + d*x) * x^2");
    println!(
        "  params: n={N}, slots={M}, base2k={BASE2K}, ct_k={CT_K}, prec_ct=({}, {}), prec_pt=({}, {})",
        PREC_CT.log_delta, PREC_CT.log_budget, PREC_PT.log_delta, PREC_PT.log_budget
    );

    let module = Module::<BakcendImpl>::new(N as u64);
    let encoder = Encoder::<f64>::new(M)?;

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_layout());
    sk_raw.fill_ternary_hw(HW, &mut source_xs);

    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    println!("  prepared secret key");
    let ct_infos = module.ckks_ciphertext_alloc_from_infos(&glwe_layout());
    let scratch_bytes = module
        .ckks_all_ops_tmp_bytes(&ct_infos, &tsk_layout(), &PREC_PT)
        .max(module.glwe_normalize_tmp_bytes());
    let mut scratch = ScratchOwned::<BakcendImpl>::alloc(scratch_bytes);
    println!("  scratch bytes: {scratch_bytes}");

    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_layout());
    {
        let mut scratch_local = scratch.borrow();
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk_raw,
            &tsk_layout(),
            &mut source_xa,
            &mut source_xe,
            &mut scratch_local,
        );
    }

    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
    {
        let mut scratch_local = scratch.borrow();
        module.prepare_tensor_key(&mut tsk_prepared, &tsk, &mut scratch_local);
    }
    println!("  prepared tensor key");

    Ok(SetupArtifacts {
        module,
        encoder,
        sk,
        tsk_prepared,
        scratch,
    })
}

/// Phase 2: encoding.
///
/// This phase builds the cleartext input slots `x`, defines the polynomial
/// coefficients `(a, b, c, d)`, and encodes `x` from slot form into a quantized
/// ZNX plaintext suitable for encryption.
///
/// The constant coefficients are prepared as host-side ZNX constants and later
/// uploaded or consumed directly by the backend.
fn encoding(setup: &SetupArtifacts) -> Result<EncodingArtifacts> {
    print_phase("encoding");

    let x_re: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).cos())
        .collect();
    let x_im: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).sin())
        .collect();

    let coeffs = PolynomialCoeffs {
        a: quantize_coeff(0.125, -0.625),
        b: quantize_coeff(0.625, -0.125),
        c: quantize_coeff(-0.375, 0.25),
        d: quantize_coeff(0.3125, -0.1875),
    };
    println!("  coefficients:");
    println!("    a = {}", format_complex(coeffs.a.0, coeffs.a.1, 4));
    println!("    b = {}", format_complex(coeffs.b.0, coeffs.b.1, 4));
    println!("    c = {}", format_complex(coeffs.c.0, coeffs.c.1, 4));
    println!("    d = {}", format_complex(coeffs.d.0, coeffs.d.1, 4));

    let cst_a = encode_const_reim(&setup.module, Some(coeffs.a.0), Some(coeffs.a.1), BASE2K, PREC_PT)?;
    let cst_b = encode_const_reim(&setup.module, Some(coeffs.b.0), Some(coeffs.b.1), BASE2K, PREC_PT)?;
    let cst_c = encode_const_reim(&setup.module, Some(coeffs.c.0), Some(coeffs.c.1), BASE2K, PREC_PT)?;
    let cst_d = encode_const_reim(&setup.module, Some(coeffs.d.0), Some(coeffs.d.1), BASE2K, PREC_PT)?;
    let mut pt_znx = setup.module.ckks_pt_vec_znx_alloc(BASE2K.into(), PREC_CT);
    setup.encoder.encode_reim(&mut pt_znx, &x_re, &x_im)?;
    print_pt_meta("encoded plaintext x", &pt_znx);
    println!("  slot[0] cleartext = {}", format_complex(x_re[0], x_im[0], 6));

    Ok(EncodingArtifacts {
        x_re,
        x_im,
        coeffs,
        cst_a,
        cst_b,
        cst_c,
        cst_d,
        pt_znx,
    })
}

/// Phase 3: encryption.
///
/// This phase encrypts the already-quantized plaintext vector `x` into a CKKS
/// ciphertext. The resulting metadata reflects:
/// - the plaintext decimal precision carried into the ciphertext
/// - the remaining homomorphic capacity left after encryption noise is budgeted
fn encryption(setup: &mut SetupArtifacts, encoding: &EncodingArtifacts) -> Result<EncryptionArtifacts> {
    print_phase("encryption");

    let mut ct_x = setup.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    let mut source_xa = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);

    {
        let mut scratch = setup.scratch.borrow();
        setup.module.ckks_encrypt_sk(
            &mut ct_x,
            &encoding.pt_znx,
            &setup.sk,
            &glwe_layout(),
            &mut source_xa,
            &mut source_xe,
            &mut scratch,
        )?;
    }

    print_ct_meta("ciphertext x", &ct_x);

    Ok(EncryptionArtifacts { ct_x })
}

/// Phase 4: evaluation.
///
/// This phase evaluates the polynomial
///
/// `p(x) = (a + b*x) + (c + d*x) * x^2`
///
/// homomorphically.
///
/// The metadata prints are the main thing to watch:
/// - `x^2` consumes one `log_delta` chunk of homomorphic capacity
/// - `ckks_compact_limbs` trims storage after each non-linear step
/// - unsafe (without-normalization) add ops are used for intermediate linear
///   steps; limbs are only K-normalized at the final step or just before a
///   ct-ct multiply that requires normalized inputs
/// - `glwe_normalize` is applied before ct-ct multiplication
/// - the final step fuses `+= b*x` via `ckks_mul_add_pt_const_znx_into`, which
///   normalizes the output as the last operation
fn evaluation(
    setup: &mut SetupArtifacts,
    encoding: &EncodingArtifacts,
    encryption: &EncryptionArtifacts,
) -> Result<EvaluationArtifacts> {
    print_phase("evaluation");
    print_ct_meta("input x", &encryption.ct_x);

    println!("  -> square x");
    let mut ct_x2 = setup
        .module
        .ckks_ciphertext_alloc(BASE2K.into(), encryption.ct_x.log_budget().into());
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_square_into(&mut ct_x2, &encryption.ct_x, &setup.tsk_prepared, &mut scratch)?;
    }
    print_ct_meta("x^2", &ct_x2);
    println!("  -> compact limbs after square");
    setup.module.ckks_compact_limbs(&mut ct_x2)?;
    print_ct_meta("x^2 compacted", &ct_x2);

    let linear_k = encryption.ct_x.effective_k() - PREC_PT.log_delta;

    println!("  -> build right branch: c + d*x (unsafe add, normalize before ct-ct mul)");
    let mut right_linear = setup.module.ckks_ciphertext_alloc(BASE2K.into(), linear_k.into());
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_mul_pt_const_znx_into(&mut right_linear, &encryption.ct_x, &encoding.cst_d, &mut scratch)?;
    }
    print_ct_meta("d * x", &right_linear);
    unsafe {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_add_pt_const_znx_assign_unsafe(&mut right_linear, 0, &encoding.cst_c, 0, &mut scratch)?;
    }
    print_ct_meta("c + d * x (not normalized)", &right_linear);

    println!("  -> normalize right branch before ct-ct multiply");
    {
        let mut scratch = setup.scratch.borrow();
        setup.module.glwe_normalize_assign(
            &mut <CKKSCiphertext<Vec<u8>> as GLWEToBackendMut<BakcendImpl>>::to_backend_mut(&mut right_linear),
            &mut scratch,
        );
    }
    print_ct_meta("c + d * x normalized", &right_linear);

    println!("  -> multiply right branch by x^2");
    let right_branch_k = ct_x2.effective_k() - ct_x2.log_delta();
    let mut right_branch = setup.module.ckks_ciphertext_alloc(BASE2K.into(), right_branch_k.into());
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_mul_into(&mut right_branch, &right_linear, &ct_x2, &setup.tsk_prepared, &mut scratch)?;
    }
    print_ct_meta("(c + d * x) * x^2", &right_branch);
    println!("  -> compact limbs after ct-ct multiply");
    setup.module.ckks_compact_limbs(&mut right_branch)?;
    print_ct_meta("(c + d * x) * x^2 compacted", &right_branch);

    println!("  -> final assembly: right_branch + a (unsafe), then += b*x (normalizing)");
    let mut poly = setup
        .module
        .ckks_ciphertext_alloc(BASE2K.into(), right_branch.effective_k().into());
    unsafe {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_add_pt_const_znx_into_unsafe(&mut poly, &right_branch, 0, &encoding.cst_a, 0, &mut scratch)?;
    }
    print_ct_meta("right_branch + a (not normalized)", &poly);
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_mul_add_pt_const_znx_into(&mut poly, &encryption.ct_x, &encoding.cst_b, &mut scratch)?;
    }
    print_ct_meta("final polynomial", &poly);

    Ok(EvaluationArtifacts { poly })
}

/// Phase 5: decryption.
///
/// This phase decrypts the final ciphertext back into a ZNX plaintext, then
/// decodes it to RNX and finally back to slot-domain complex values.
///
/// At the end of this phase we have ordinary floating-point vectors that can
/// be compared to the direct cleartext polynomial evaluation.
fn decryption(setup: &mut SetupArtifacts, evaluation: &EvaluationArtifacts) -> Result<DecryptionArtifacts> {
    print_phase("decryption");

    let mut pt_znx = setup.module.ckks_pt_vec_znx_alloc_from_infos(&evaluation.poly);
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_decrypt(&mut pt_znx, &evaluation.poly, &setup.sk, &mut scratch)?;
    }
    print_pt_meta("decrypted plaintext", &pt_znx);

    let mut have_re = vec![0.0; M];
    let mut have_im = vec![0.0; M];
    setup.encoder.decode_reim(&pt_znx, &mut have_re, &mut have_im)?;
    println!("  slot[0] decrypted = {}", format_complex(have_re[0], have_im[0], 6));

    Ok(DecryptionArtifacts { have_re, have_im })
}

/// Phase 6: verification.
///
/// This phase evaluates the same cubic polynomial directly in cleartext and compares
/// it against the decrypted result. The final prints summarize:
/// - the remaining homomorphic capacity of the output ciphertext
/// - the worst absolute error on the real and imaginary channels
/// - one representative slot comparison
fn verification(encoding: &EncodingArtifacts, evaluation: &EvaluationArtifacts, decryption: &DecryptionArtifacts) -> Result<()> {
    print_phase("verification");

    let want_re: Vec<f64> = (0..M)
        .map(|j| {
            let xr = encoding.x_re[j];
            let xi = encoding.x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            let left_re = encoding.coeffs.a.0 + encoding.coeffs.b.0 * xr - encoding.coeffs.b.1 * xi;
            let right_re = encoding.coeffs.c.0 + encoding.coeffs.d.0 * xr - encoding.coeffs.d.1 * xi;
            let right_im = encoding.coeffs.c.1 + encoding.coeffs.d.0 * xi + encoding.coeffs.d.1 * xr;
            left_re + right_re * x2r - right_im * x2i
        })
        .collect();
    let want_im: Vec<f64> = (0..M)
        .map(|j| {
            let xr = encoding.x_re[j];
            let xi = encoding.x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            let left_im = encoding.coeffs.a.1 + encoding.coeffs.b.0 * xi + encoding.coeffs.b.1 * xr;
            let right_re = encoding.coeffs.c.0 + encoding.coeffs.d.0 * xr - encoding.coeffs.d.1 * xi;
            let right_im = encoding.coeffs.c.1 + encoding.coeffs.d.0 * xi + encoding.coeffs.d.1 * xr;
            left_im + right_re * x2i + right_im * x2r
        })
        .collect();

    let err_re = max_err(&decryption.have_re, &want_re);
    let err_im = max_err(&decryption.have_im, &want_im);

    print_ct_meta("verified polynomial", &evaluation.poly);
    println!("  max error          re={err_re:.3e}  im={err_im:.3e}");
    println!(
        "  slot[0] decrypted  {}",
        format_complex(decryption.have_re[0], decryption.have_im[0], 6)
    );
    println!("  slot[0] expected   {}", format_complex(want_re[0], want_im[0], 6));

    assert!(err_re < 1e-4);
    assert!(err_im < 1e-4);
    println!("  status: PASS");

    Ok(())
}

/// Runs the six CKKS phases in order.
fn main() -> Result<()> {
    let mut setup_artifacts = setup()?;
    let encoding_artifacts = encoding(&setup_artifacts)?;
    let encryption_artifacts = encryption(&mut setup_artifacts, &encoding_artifacts)?;
    let evaluation_artifacts = evaluation(&mut setup_artifacts, &encoding_artifacts, &encryption_artifacts)?;
    let decryption_artifacts = decryption(&mut setup_artifacts, &evaluation_artifacts)?;
    verification(&encoding_artifacts, &evaluation_artifacts, &decryption_artifacts)
}
