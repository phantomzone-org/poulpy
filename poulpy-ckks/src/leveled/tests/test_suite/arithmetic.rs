use super::CKKSTestParams;
use crate::{
    encoding::classical::{decode, encode, encode_tmp_bytes},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::{
        arithmetic::{
            add_const_ct, add_const_ct_inplace, add_ct_ct, add_ct_ct_inplace, add_pt_ct, add_pt_ct_inplace, neg_ct,
            neg_ct_inplace, sub_const_ct, sub_const_ct_inplace, sub_ct_ct, sub_ct_ct_inplace, sub_pt_ct, sub_pt_ct_inplace,
        },
        encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWESub, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegate, VecZnxNegateInplace,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

pub trait ArithBounds: Backend {}
impl<BE: Backend> ArithBounds for BE {}

pub trait CodecBounds: Backend<ScalarPrep = f64, ScalarBig = i64> {}
impl<BE: Backend<ScalarPrep = f64, ScalarBig = i64>> CodecBounds for BE {}

struct SetupResult<BE: Backend> {
    sk: GLWESecretPrepared<Vec<u8>, BE>,
    re1: Vec<f64>,
    im1: Vec<f64>,
    re2: Vec<f64>,
    im2: Vec<f64>,
}

fn setup<BE: ArithBounds>(module: &Module<BE>, params: CKKSTestParams) -> SetupResult<BE>
where
    Module<BE>: ModuleN + GLWESecretPreparedFactory<BE>,
{
    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);

    let glwe_infos = GLWELayout {
        n: Degree(n as u32),
        base2k,
        k,
        rank: Rank(1),
    };

    let mut source_xs = Source::new([0u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let re1 = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im1 = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();
    let re2 = (0..m).map(|i| 0.3 * (1.0 - (i as f64) / (m as f64))).collect();
    let im2 = (0..m).map(|i| -0.2 * (i as f64) / (m as f64)).collect();

    SetupResult {
        sk: sk_prepared,
        re1,
        im1,
        re2,
        im2,
    }
}

fn alloc_scratches<BE: ArithBounds, CE: CodecBounds>(
    module: &Module<BE>,
    codec: &Module<CE>,
    params: CKKSTestParams,
) -> (ScratchOwned<BE>, ScratchOwned<CE>)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE>,
    Module<CE>: ModuleN + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE>,
{
    assert_eq!(module.n(), codec.n(), "module/codec ring degree mismatch");

    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);

    let scratch_be = ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(module, &ct).max(decrypt_tmp_bytes(module, &ct)));
    let scratch_ce = ScratchOwned::<CE>::alloc(encode_tmp_bytes(codec));
    (scratch_be, scratch_ce)
}

fn encrypt<BE: ArithBounds, CE: CodecBounds>(
    module: &Module<BE>,
    codec: &Module<CE>,
    params: CKKSTestParams,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    re: &[f64],
    im: &[f64],
    scratch_be: &mut ScratchOwned<BE>,
    scratch_ce: &mut ScratchOwned<CE>,
    source_xa: &mut Source,
    source_xe: &mut Source,
) -> CKKSCiphertext<Vec<u8>>
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE>,
    Module<CE>: ModuleN + VecZnxDftAlloc<CE> + VecZnxIdftApplyConsume<CE> + VecZnxBigNormalize<CE> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    encode(codec, &mut pt, re, im, scratch_ce.borrow());
    encrypt_sk(module, &mut ct, &pt, sk, source_xa, source_xe, scratch_be.borrow());
    ct
}

fn decrypt_decode<BE: ArithBounds, CE: CodecBounds>(
    module: &Module<BE>,
    codec: &Module<CE>,
    params: CKKSTestParams,
    ct: &CKKSCiphertext<Vec<u8>>,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    scratch_be: &mut ScratchOwned<BE>,
) -> (Vec<f64>, Vec<f64>)
where
    Module<BE>: ModuleN + GLWEDecrypt<BE>,
    Module<CE>: ModuleN + VecZnxDftAlloc<CE> + VecZnxDftApply<CE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    decrypt(module, &mut pt_out, ct, sk, scratch_be.borrow());
    decode(codec, &pt_out)
}

fn tol(n: usize, log_delta: u32) -> f64 {
    2.0 * (n as f64) * 6.0 * SIGMA / (1u64 << log_delta) as f64
}

pub fn test_add_ct_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    let ct2 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re2,
        &im2,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([3u8; 32]),
        &mut Source::new([4u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    add_ct_ct(module, &mut ct_res, &ct1, &ct2);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = 2.0 * tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] + re2[j])).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] + re2[j]
        );
        assert!(
            (im_out[j] - (im1[j] + im2[j])).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] + im2[j]
        );
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    add_ct_ct_inplace(module, &mut ct_ip, &ct2);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_pt_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut pt2 = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    encode(codec, &mut pt2, &re2, &im2, scratch_ce.borrow());

    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    add_pt_ct(module, &mut ct_res, &ct1, &pt2);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] + re2[j])).abs() < t, "re[{j}]");
        assert!((im_out[j] - (im1[j] + im2[j])).abs() < t, "im[{j}]");
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    add_pt_ct_inplace(module, &mut ct_ip, &pt2);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_const_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, .. } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let c_re = 0.1;
    let c_im = -0.05;

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    add_const_ct(module, &mut ct_res, &ct1, c_re, c_im);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] + c_re)).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] + c_re
        );
        assert!(
            (im_out[j] - (im1[j] + c_im)).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] + c_im
        );
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    add_const_ct_inplace(module, &mut ct_ip, c_re, c_im);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + c_re)).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + c_im)).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_sub_ct_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd + GLWESub,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    let ct2 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re2,
        &im2,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([3u8; 32]),
        &mut Source::new([4u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    sub_ct_ct(module, &mut ct_res, &ct1, &ct2);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = 2.0 * tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] - re2[j])).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] - re2[j]
        );
        assert!(
            (im_out[j] - (im1[j] - im2[j])).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] - im2[j]
        );
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    sub_ct_ct_inplace(module, &mut ct_ip, &ct2);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] - re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] - im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_sub_pt_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd + GLWESub,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut pt2 = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    encode(codec, &mut pt2, &re2, &im2, scratch_ce.borrow());

    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    sub_pt_ct(module, &mut ct_res, &ct1, &pt2);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] - re2[j])).abs() < t, "re[{j}]");
        assert!((im_out[j] - (im1[j] - im2[j])).abs() < t, "im[{j}]");
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    sub_pt_ct_inplace(module, &mut ct_ip, &pt2);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] - re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] - im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_sub_const_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd + GLWESub,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, .. } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let c_re = 0.1;
    let c_im = -0.05;

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    sub_const_ct(module, &mut ct_res, &ct1, c_re, c_im);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] - c_re)).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] - c_re
        );
        assert!(
            (im_out[j] - (im1[j] - c_im)).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] - c_im
        );
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    sub_const_ct_inplace(module, &mut ct_ip, c_re, c_im);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] - c_re)).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] - c_im)).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_neg_ct<BE: ArithBounds, CE: CodecBounds>(module: &Module<BE>, codec: &Module<CE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
        + VecZnxNegate
        + VecZnxNegateInplace,
    Module<CE>: ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxDftApply<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, .. } = setup(module, params);
    let (mut scratch_be, mut scratch_ce) = alloc_scratches(module, codec, params);

    let ct1 = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    neg_ct(module, &mut ct_res, &ct1);
    let (re_out, im_out) = decrypt_decode(module, codec, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!((re_out[j] + re1[j]).abs() < t, "re[{j}]: {} vs {}", re_out[j], -re1[j]);
        assert!((im_out[j] + im1[j]).abs() < t, "im[{j}]: {} vs {}", im_out[j], -im1[j]);
    }

    let mut ct_ip = encrypt(
        module,
        codec,
        params,
        &sk,
        &re1,
        &im1,
        &mut scratch_be,
        &mut scratch_ce,
        &mut Source::new([1u8; 32]),
        &mut Source::new([2u8; 32]),
    );
    neg_ct_inplace(module, &mut ct_ip);
    let (re_ip, im_ip) = decrypt_decode(module, codec, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] + re1[j]).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] + im1[j]).abs() < t, "inplace im[{j}]");
    }
}
