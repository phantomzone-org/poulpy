use super::CKKSTestParams;
use crate::{
    encoding::classical::{decode, encode},
    layouts::{
        ciphertext::CKKSCiphertext,
        keys::{tensor_key::CKKSTensorKey, tensor_key_prepared::CKKSTensorKeyPrepared},
        plaintext::CKKSPlaintext,
    },
    leveled::{
        encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
        operations::mul::{mul, mul_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWETensorKey, GLWETensorKeyPreparedFactory, LWEInfos,
        Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

pub fn test_mul<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEShift<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let degree = Degree(n as u32);

    let glwe_infos = GLWELayout {
        n: degree,
        base2k,
        k,
        rank: Rank(1),
    };

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let tsk_layout = CKKSTensorKey::layout(degree, base2k, k);
    let mut tsk = CKKSTensorKey::alloc(degree, base2k, k);

    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(module, &CKKSCiphertext::alloc(degree, base2k, k, params.log_delta))
            .max(decrypt_tmp_bytes(
                module,
                &CKKSCiphertext::alloc(degree, base2k, k, params.log_delta),
            ))
            .max(module.prepare_tensor_key_tmp_bytes(&tsk_layout))
            .max(GLWETensorKey::<Vec<u8>>::encrypt_sk_tmp_bytes(module, &tsk_layout)),
    );

    tsk.encrypt_sk(module, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut tsk_prepared = CKKSTensorKeyPrepared::alloc(module, degree, base2k, k);
    tsk_prepared.prepare(module, &tsk, scratch.borrow());

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, params.log_delta);
    let mul_scratch = mul_tmp_bytes(module, &ct_tmp, &ct_tmp, &tsk_prepared);
    scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(module, &ct_tmp)
            .max(decrypt_tmp_bytes(module, &ct_tmp))
            .max(mul_scratch),
    );

    let re1: Vec<f64> = (0..m).map(|i| 0.25 * (i as f64) / (m as f64) - 0.125).collect();
    let im1: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();
    let re2: Vec<f64> = (0..m).map(|i| 0.2 * (1.0 - (i as f64) / (m as f64))).collect();
    let im2: Vec<f64> = (0..m).map(|i| -0.1 * (i as f64) / (m as f64)).collect();

    let mut pt1 = CKKSPlaintext::alloc(degree, base2k, k, params.log_delta);
    let mut pt2 = CKKSPlaintext::alloc(degree, base2k, k, params.log_delta);
    encode(&mut pt1, &re1, &im1);
    encode(&mut pt2, &re2, &im2);

    let mut ct1 = CKKSCiphertext::alloc(degree, base2k, k, params.log_delta);
    let mut ct2 = CKKSCiphertext::alloc(degree, base2k, k, params.log_delta);
    encrypt_sk(
        module,
        &mut ct1,
        &pt1,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    encrypt_sk(
        module,
        &mut ct2,
        &pt2,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ct_res = CKKSCiphertext::alloc(degree, base2k, k, params.log_delta);
    mul(module, &mut ct_res, &ct1, &ct2, &tsk_prepared, scratch.borrow());

    assert_eq!(
        ct_res.log_delta, params.log_delta,
        "log_delta mismatch: {} != {}",
        ct_res.log_delta, params.log_delta
    );

    let mut pt_out = CKKSPlaintext::alloc(degree, base2k, ct_res.inner.k(), params.log_delta);
    decrypt(module, &mut pt_out, &ct_res, &sk_prepared, scratch.borrow());
    let (re_out, im_out) = decode(&pt_out);

    let mut max_err_re: f64 = 0.0;
    let mut max_err_im: f64 = 0.0;
    for j in 0..m {
        let want_re = re1[j] * re2[j] - im1[j] * im2[j];
        let want_im = re1[j] * im2[j] + im1[j] * re2[j];
        max_err_re = max_err_re.max((re_out[j] - want_re).abs());
        max_err_im = max_err_im.max((im_out[j] - want_im).abs());
    }

    let prec_re = -(max_err_re.log2());
    let prec_im = -(max_err_im.log2());
    let min_prec = prec_re.min(prec_im);

    assert!(
        min_prec > 5.0,
        "precision too low: {min_prec:.1} bits (re={prec_re:.1}, im={prec_im:.1}, max_err_re={max_err_re}, max_err_im={max_err_im})"
    );
}
