use anyhow::Result;
use poulpy_ckks::{
    CKKS, CKKSInfos,
    encoding::Encoder,
    layouts::{CKKSCiphertext, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextVecRnx, CKKSPlaintextVecZnx},
    leveled::{
        encryption::{CKKSDecrypt, CKKSEncrypt},
        operations::{add::CKKSAddOps, mul::CKKSMulOps},
    },
};
use poulpy_core::{
    EncryptionLayout, GLWEShift, GLWETensorKeyEncryptSk,
    layouts::{
        GLWELayout, GLWESecret, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, Rank,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataRef, DeviceBuf, Module, Scratch, ScratchOwned},
    oep::HalImpl,
    source::Source,
};

const N: usize = 256;
const M: usize = N / 2;
const BASE2K: usize = 52;
const CT_K: usize = 8 * BASE2K + 1;
const HW: usize = 192;
const DSIZE: usize = 1;
const PREC: CKKS = CKKS {
    log_decimal: 40,
    log_hom_rem: 30,
};

fn glwe_layout() -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    })
    .unwrap()
}

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

fn encode_pt_znx<BE: Backend>(
    encoder: &Encoder<FFT64Ref>,
    re: &[f64],
    im: &[f64],
    prec: CKKS,
) -> Result<CKKSPlaintextVecZnx<Vec<u8>>> {
    let mut pt_rnx = CKKSPlaintextVecRnx::<f64>::alloc(N)?;
    encoder.encode_reim(&mut pt_rnx, re, im)?;

    let mut pt_znx = CKKSPlaintextVecZnx::alloc(N.into(), BASE2K.into(), prec);
    pt_rnx.to_znx::<BE>(&mut pt_znx)?;
    Ok(pt_znx)
}

fn constant_rnx(value: (f64, f64)) -> CKKSPlaintextCstRnx<f64> {
    CKKSPlaintextCstRnx::new(Some(value.0), Some(value.1))
}

fn encrypt<BE: Backend + HalImpl<BE>>(
    module: &Module<BE>,
    sk: &GLWESecretPrepared<DeviceBuf<BE>, BE>,
    pt: &CKKSPlaintextVecZnx<impl DataRef>,
    scratch: &mut Scratch<BE>,
) -> Result<CKKSCiphertext<Vec<u8>>>
where
    Module<BE>: CKKSEncrypt<BE>,
{
    let mut ct = CKKSCiphertext::alloc(N.into(), CT_K.into(), BASE2K.into());
    let mut source_xa = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
    module.ckks_encrypt_sk(&mut ct, pt, sk, &glwe_layout(), &mut source_xa, &mut source_xe, scratch)?;
    Ok(ct)
}

fn decrypt_decode<BE: Backend + HalImpl<BE>>(
    module: &Module<BE>,
    encoder: &Encoder<FFT64Ref>,
    sk: &GLWESecretPrepared<DeviceBuf<BE>, BE>,
    ct: &CKKSCiphertext<impl DataRef>,
    scratch: &mut Scratch<BE>,
) -> Result<(Vec<f64>, Vec<f64>)>
where
    Module<BE>: CKKSDecrypt<BE>,
{
    let mut pt_znx = CKKSPlaintextVecZnx::alloc(
        N.into(),
        ct.base2k(),
        CKKS {
            log_decimal: ct.log_decimal(),
            log_hom_rem: 0,
        },
    );
    module.ckks_decrypt(&mut pt_znx, ct, sk, scratch)?;

    let mut pt_rnx = CKKSPlaintextVecRnx::<f64>::alloc(N)?;
    pt_rnx.decode_from_znx::<BE>(&pt_znx)?;

    let mut re = vec![0.0; M];
    let mut im = vec![0.0; M];
    encoder.decode_reim(&pt_rnx, &mut re, &mut im)?;
    Ok((re, im))
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

fn main() -> Result<()> {
    let module = Module::<NTT120Ref>::new(N as u64);
    let encoder = Encoder::<FFT64Ref>::new(M)?;

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = GLWESecret::alloc_from_infos(&glwe_layout());
    sk_raw.fill_ternary_hw(HW, &mut source_xs);

    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    let mut tsk = GLWETensorKey::alloc_from_infos(&tsk_layout());
    let mut scratch = ScratchOwned::<NTT120Ref>::alloc(
        module
            .ckks_encrypt_sk_tmp_bytes(&glwe_layout())
            .max(module.ckks_decrypt_tmp_bytes(&glwe_layout()))
            .max(module.prepare_tensor_key_tmp_bytes(&tsk_layout()))
            .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_layout()))
            .max(module.ckks_mul_tmp_bytes(&glwe_layout(), &tsk_layout()))
            .max(module.ckks_square_tmp_bytes(&glwe_layout(), &tsk_layout()))
            .max(module.ckks_mul_const_tmp_bytes(&glwe_layout(), &glwe_layout(), &PREC))
            .max(module.glwe_shift_tmp_bytes()),
    );

    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk_raw,
        &tsk_layout(),
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    let mut tsk_prepared: GLWETensorKeyPrepared<DeviceBuf<NTT120Ref>, NTT120Ref> =
        module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
    module.prepare_tensor_key(&mut tsk_prepared, &tsk, scratch.borrow());

    let x_re: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).cos())
        .collect();
    let x_im: Vec<f64> = (0..M)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / M as f64).sin())
        .collect();

    let a = (0.125, -0.0625);
    let b = (0.625, -0.125);
    let c = (-0.375, 0.25);

    let pt_x = encode_pt_znx::<NTT120Ref>(&encoder, &x_re, &x_im, PREC)?;
    let ct_x = encrypt(&module, &sk, &pt_x, scratch.borrow())?;

    let mut ct_x2 = CKKSCiphertext::alloc(N.into(), ct_x.log_hom_rem().into(), BASE2K.into());
    module.ckks_square(&mut ct_x2, &ct_x, &tsk_prepared, scratch.borrow())?;

    let cst_b = constant_rnx(b);
    let cst_c = constant_rnx(c);
    let cst_a = constant_rnx(a);

    let mut term_bx = CKKSCiphertext::alloc(N.into(), ct_x.log_hom_rem().into(), BASE2K.into());
    module.ckks_mul_pt_const_rnx(&mut term_bx, &ct_x, &cst_b, PREC, scratch.borrow())?;

    let mut term_cx2 = CKKSCiphertext::alloc(N.into(), ct_x2.log_hom_rem().into(), BASE2K.into());
    module.ckks_mul_pt_const_rnx(&mut term_cx2, &ct_x2, &cst_c, PREC, scratch.borrow())?;

    let mut poly = CKKSCiphertext::alloc(N.into(), term_cx2.effective_k().into(), BASE2K.into());
    module.ckks_add(&mut poly, &term_bx, &term_cx2, scratch.borrow())?;
    module.ckks_add_pt_const_rnx_inplace(&mut poly, &cst_a, PREC, scratch.borrow())?;

    let (have_re, have_im) = decrypt_decode(&module, &encoder, &sk, &poly, scratch.borrow())?;

    let want_re: Vec<f64> = (0..M)
        .map(|j| {
            let xr = x_re[j];
            let xi = x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            a.0 + b.0 * xr - b.1 * xi + c.0 * x2r - c.1 * x2i
        })
        .collect();
    let want_im: Vec<f64> = (0..M)
        .map(|j| {
            let xr = x_re[j];
            let xi = x_im[j];
            let x2r = xr * xr - xi * xi;
            let x2i = 2.0 * xr * xi;
            a.1 + b.0 * xi + b.1 * xr + c.0 * x2i + c.1 * x2r
        })
        .collect();

    let err_re = max_err(&have_re, &want_re);
    let err_im = max_err(&have_im, &want_im);

    println!("max error: re={err_re:.3e}, im={err_im:.3e}");
    println!(
        "slot 0: have=({:.6}, {:.6}) want=({:.6}, {:.6})",
        have_re[0], have_im[0], want_re[0], want_im[0]
    );

    assert!(err_re < 1e-4);
    assert!(err_im < 1e-4);

    Ok(())
}
