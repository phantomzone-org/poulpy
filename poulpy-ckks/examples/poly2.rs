//! End-to-end CKKS example for evaluating a quadratic polynomial
//!
//! This example evaluates
//!
//! `p(x) = a + b*x + c*x^2`
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
    layouts::{CKKSCiphertext, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextVecRnx, CKKSPlaintextVecZnx},
    leveled::{
        encryption::{CKKSDecrypt, CKKSEncrypt},
        operations::{add::CKKSAddOps, mul::CKKSMulOps},
        tmp_bytes::CKKSAllOpsTmpBytes,
    },
};
use poulpy_core::{
    EncryptionLayout, GLWETensorKeyEncryptSk,
    layouts::{
        GLWELayout, GLWESecret, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_ref::NTT120Ref;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DeviceBuf, Module, ScratchOwned},
    source::Source,
};

pub type BakcendImpl = NTT120Ref;

type SecretKeyPrepared = GLWESecretPrepared<DeviceBuf<BakcendImpl>, BakcendImpl>;
type TensorKeyPrepared = GLWETensorKeyPrepared<DeviceBuf<BakcendImpl>, BakcendImpl>;

const N: usize = 4096;
const M: usize = N / 2;
const BASE2K: usize = 52;
const CT_K: usize = 75;
const HW: usize = 192;
const DSIZE: usize = 1;
const PREC_CT: CKKSMeta = CKKSMeta {
    log_decimal: 30,
    log_hom_rem: 5,
};
const PREC_PT: CKKSMeta = CKKSMeta {
    log_decimal: 8,
    log_hom_rem: 0,
};

/// Long-lived objects prepared during setup and reused by the later phases.
struct SetupArtifacts {
    module: Module<BakcendImpl>,
    encoder: Encoder<f64>,
    sk: SecretKeyPrepared,
    tsk_prepared: TensorKeyPrepared,
    scratch: ScratchOwned<BakcendImpl>,
}

/// Complex coefficients of the polynomial `a + b*x + c*x^2`.
struct PolynomialCoeffs {
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
}

/// Inputs prepared by the encoding phase.
///
/// This includes both the encoded slot vector `x` and the plaintext constants
/// used later during evaluation.
struct EncodingArtifacts {
    x_re: Vec<f64>,
    x_im: Vec<f64>,
    coeffs: PolynomialCoeffs,
    cst_a: CKKSPlaintextCstRnx<f64>,
    cst_b: CKKSPlaintextCstRnx<f64>,
    cst_c: CKKSPlaintextCstRnx<f64>,
    pt_znx: CKKSPlaintextVecZnx<Vec<u8>>,
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

/// Prints the semantic and storage metadata of a CKKS ciphertext.
fn print_ct_meta(label: &str, ct: &CKKSCiphertext<Vec<u8>>) {
    println!(
        "{label}: log_decimal={}, log_hom_rem={}, effective_k={}, max_k={}",
        ct.log_decimal(),
        ct.log_hom_rem(),
        ct.effective_k(),
        ct.max_k().as_usize()
    );
}

/// Prints the semantic and storage metadata of a CKKS plaintext.
fn print_pt_meta(label: &str, pt: &CKKSPlaintextVecZnx<Vec<u8>>) {
    println!(
        "{label}: log_decimal={}, log_hom_rem={}, effective_k={}, max_k={}",
        pt.log_decimal(),
        pt.log_hom_rem(),
        pt.effective_k(),
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
    println!("== setup ==");
    println!(
        "parameters: n={N}, slots={M}, base2k={BASE2K}, ct_k={CT_K}, prec_ct=({}, {}), prec_pt=({}, {})",
        PREC_CT.log_decimal, PREC_CT.log_hom_rem, PREC_PT.log_decimal, PREC_PT.log_hom_rem
    );

    let module = Module::<BakcendImpl>::new(N as u64);
    let encoder = Encoder::<f64>::new(M)?;

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = GLWESecret::alloc_from_infos(&glwe_layout());
    sk_raw.fill_ternary_hw(HW, &mut source_xs);

    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    let scratch_bytes = module.ckks_all_ops_tmp_bytes(&glwe_layout(), &tsk_layout(), &PREC_PT);
    let mut scratch = ScratchOwned::<BakcendImpl>::alloc(scratch_bytes);
    println!("scratch bytes allocated: {scratch_bytes}");

    let mut tsk = GLWETensorKey::alloc_from_infos(&tsk_layout());
    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk_raw,
        &tsk_layout(),
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
    module.prepare_tensor_key(&mut tsk_prepared, &tsk, scratch.borrow());
    println!("secret key prepared");
    println!("tensor key prepared");

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
/// coefficients `(a, b, c)`, and encodes `x` from slot form into a quantized
/// ZNX plaintext suitable for encryption.
///
/// The constant coefficients stay as RNX scalars because the evaluation APIs
/// can quantize them on demand with the correct metadata.
fn encoding(setup: &SetupArtifacts) -> Result<EncodingArtifacts> {
    println!("== encoding ==");

    let x_re: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).cos())
        .collect();
    let x_im: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).sin())
        .collect();

    let coeffs = PolynomialCoeffs {
        a: (0.125, -0.625),
        b: (0.625, -0.125),
        c: (-0.375, 0.25),
    };
    println!(
        "polynomial coefficients: a=({:.3}, {:.3}), b=({:.3}, {:.3}), c=({:.3}, {:.3})",
        coeffs.a.0, coeffs.a.1, coeffs.b.0, coeffs.b.1, coeffs.c.0, coeffs.c.1
    );

    let cst_a = CKKSPlaintextCstRnx::new(Some(coeffs.a.0), Some(coeffs.a.1));
    let cst_b = CKKSPlaintextCstRnx::new(Some(coeffs.b.0), Some(coeffs.b.1));
    let cst_c = CKKSPlaintextCstRnx::new(Some(coeffs.c.0), Some(coeffs.c.1));

    let mut pt_rnx = CKKSPlaintextVecRnx::<f64>::alloc(N)?;
    setup.encoder.encode_reim(&mut pt_rnx, &x_re, &x_im)?;

    let mut pt_znx = CKKSPlaintextVecZnx::alloc(N.into(), BASE2K.into(), PREC_CT);
    pt_rnx.to_znx(&mut pt_znx)?;
    print_pt_meta("encoded plaintext x", &pt_znx);
    println!("slot 0 cleartext input: x=({:.6}, {:.6})", x_re[0], x_im[0]);

    Ok(EncodingArtifacts {
        x_re,
        x_im,
        coeffs,
        cst_a,
        cst_b,
        cst_c,
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
    println!("== encryption ==");

    let mut ct_x = CKKSCiphertext::alloc(N.into(), CT_K.into(), BASE2K.into());
    let mut source_xa = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);

    setup.module.ckks_encrypt_sk(
        &mut ct_x,
        &encoding.pt_znx,
        &setup.sk,
        &glwe_layout(),
        &mut source_xa,
        &mut source_xe,
        setup.scratch.borrow(),
    )?;

    print_ct_meta("ciphertext x", &ct_x);

    Ok(EncryptionArtifacts { ct_x })
}

/// Phase 4: evaluation.
///
/// This phase evaluates the polynomial
///
/// `p(x) = a + b*x + c*x^2`
///
/// homomorphically.
///
/// The metadata prints are the main thing to watch:
/// - `x^2` consumes one `log_decimal` chunk of homomorphic capacity
/// - multiplying by a plaintext constant consumes the constant's
///   `log_decimal`
/// - adding aligned ciphertexts preserves the minimum remaining capacity
/// - adding the constant `a` does not consume extra ciphertext capacity beyond
///   alignment
fn evaluation(
    setup: &mut SetupArtifacts,
    encoding: &EncodingArtifacts,
    encryption: &EncryptionArtifacts,
) -> Result<EvaluationArtifacts> {
    println!("== evaluation ==");
    print_ct_meta("input x", &encryption.ct_x);

    println!("computing x^2");
    let mut ct_x2 = CKKSCiphertext::alloc(N.into(), encryption.ct_x.log_hom_rem().into(), BASE2K.into());
    setup
        .module
        .ckks_square(&mut ct_x2, &encryption.ct_x, &setup.tsk_prepared, setup.scratch.borrow())?;
    print_ct_meta("x^2", &ct_x2);

    println!("computing c * x^2");
    let mut term_cx2 = CKKSCiphertext::alloc(N.into(), ct_x2.effective_k().into(), BASE2K.into());
    setup
        .module
        .ckks_mul_const(&mut term_cx2, &ct_x2, &encoding.cst_c, PREC_PT, setup.scratch.borrow())?;
    print_ct_meta("c * x^2", &term_cx2);

    println!("computing b * x");
    let mut term_bx = CKKSCiphertext::alloc(N.into(), encryption.ct_x.effective_k().into(), BASE2K.into());
    setup.module.ckks_mul_pt_const_rnx(
        &mut term_bx,
        &encryption.ct_x,
        &encoding.cst_b,
        PREC_PT,
        setup.scratch.borrow(),
    )?;
    print_ct_meta("b * x", &term_bx);

    println!("adding b * x and c * x^2");
    let mut poly = CKKSCiphertext::alloc(N.into(), term_cx2.effective_k().into(), BASE2K.into());
    setup
        .module
        .ckks_add(&mut poly, &term_bx, &term_cx2, setup.scratch.borrow())?;
    print_ct_meta("b * x + c * x^2", &poly);

    println!("adding constant a");
    setup
        .module
        .ckks_add_pt_const_rnx_inplace(&mut poly, &encoding.cst_a, PREC_PT, setup.scratch.borrow())?;
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
    println!("== decryption ==");

    let mut pt_znx = CKKSPlaintextVecZnx::alloc_from_infos(&evaluation.poly);
    setup
        .module
        .ckks_decrypt(&mut pt_znx, &evaluation.poly, &setup.sk, setup.scratch.borrow())?;
    print_pt_meta("decrypted plaintext", &pt_znx);

    let mut pt_rnx = CKKSPlaintextVecRnx::<f64>::alloc(N)?;
    pt_rnx.decode_from_znx(&pt_znx)?;

    let mut have_re = vec![0.0; M];
    let mut have_im = vec![0.0; M];
    setup.encoder.decode_reim(&pt_rnx, &mut have_re, &mut have_im)?;
    println!("slot 0 decrypted value: ({:.6}, {:.6})", have_re[0], have_im[0]);

    Ok(DecryptionArtifacts { have_re, have_im })
}

/// Phase 6: verification.
///
/// This phase evaluates the same polynomial directly in cleartext and compares
/// it against the decrypted result. The final prints summarize:
/// - the remaining homomorphic capacity of the output ciphertext
/// - the worst absolute error on the real and imaginary channels
/// - one representative slot comparison
fn verification(encoding: &EncodingArtifacts, evaluation: &EvaluationArtifacts, decryption: &DecryptionArtifacts) -> Result<()> {
    println!("== verification ==");

    let want_re: Vec<f64> = (0..M)
        .map(|j| {
            let xr = encoding.x_re[j];
            let xi = encoding.x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            encoding.coeffs.a.0 + encoding.coeffs.b.0 * xr - encoding.coeffs.b.1 * xi + encoding.coeffs.c.0 * x2r
                - encoding.coeffs.c.1 * x2i
        })
        .collect();
    let want_im: Vec<f64> = (0..M)
        .map(|j| {
            let xr = encoding.x_re[j];
            let xi = encoding.x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            encoding.coeffs.a.1
                + encoding.coeffs.b.0 * xi
                + encoding.coeffs.b.1 * xr
                + encoding.coeffs.c.0 * x2i
                + encoding.coeffs.c.1 * x2r
        })
        .collect();

    let err_re = max_err(&decryption.have_re, &want_re);
    let err_im = max_err(&decryption.have_im, &want_im);

    print_ct_meta("verified polynomial", &evaluation.poly);
    println!("max error: re={err_re:.3e}, im={err_im:.3e}");
    println!(
        "slot 0: have=({:.6}, {:.6}) want=({:.6}, {:.6})",
        decryption.have_re[0], decryption.have_im[0], want_re[0], want_im[0]
    );

    assert!(err_re < 1e-4);
    assert!(err_im < 1e-4);

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
