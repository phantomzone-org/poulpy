use crate::{
    encoding::classical::{encode, encode_tmp_bytes},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::{
        arithmetic::{
            add_cleartext_ct, add_cleartext_ct_inplace, add_ct_ct, add_ct_ct_inplace, add_pt_ct, add_pt_ct_inplace,
        },
        encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

const BASE2K: u32 = 52;
const LOG_DELTA: u32 = 40;
const HW: usize = 192;

pub trait ArithBounds: Backend<ScalarPrep = f64, ScalarBig = i64> {}
impl<BE: Backend<ScalarPrep = f64, ScalarBig = i64>> ArithBounds for BE {}

struct SetupResult<BE: Backend> {
    sk: GLWESecretPrepared<Vec<u8>, BE>,
    scratch: ScratchOwned<BE>,
    re1: Vec<f64>,
    im1: Vec<f64>,
    re2: Vec<f64>,
    im2: Vec<f64>,
}

fn setup<BE: ArithBounds>(module: &Module<BE>) -> SetupResult<BE>
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);

    let glwe_infos = GLWELayout { n: Degree(n as u32), base2k, k, rank: Rank(1) };

    let mut source_xs: Source = Source::new([0u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(HW, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let ct_tmp = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let scratch_size = encode_tmp_bytes(module)
        .max(encrypt_sk_tmp_bytes(module, &ct_tmp))
        .max(decrypt_tmp_bytes(module, &ct_tmp));
    let scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let re1: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im1: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();
    let re2: Vec<f64> = (0..m).map(|i| 0.3 * (1.0 - (i as f64) / (m as f64))).collect();
    let im2: Vec<f64> = (0..m).map(|i| -0.2 * (i as f64) / (m as f64)).collect();

    SetupResult { sk: sk_prepared, scratch, re1, im1, re2, im2 }
}

fn encrypt<BE: ArithBounds>(
    module: &Module<BE>,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    re: &[f64],
    im: &[f64],
    scratch: &mut ScratchOwned<BE>,
    source_xa: &mut Source,
    source_xe: &mut Source,
) -> CKKSCiphertext<Vec<u8>>
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);
    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    encode(module, &mut pt, re, im, scratch.borrow());
    encrypt_sk(module, &mut ct, &pt, sk, source_xa, source_xe, scratch.borrow());
    ct
}

fn decrypt_decode<BE: ArithBounds>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<Vec<u8>>,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> (Vec<f64>, Vec<f64>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    use crate::encoding::classical::decode;
    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    decrypt(module, &mut pt_out, ct, sk, scratch.borrow());
    decode(module, &pt_out)
}

fn tol(n: usize) -> f64 {
    2.0 * (n as f64) * 6.0 * SIGMA / (1u64 << LOG_DELTA) as f64
}

pub fn test_add_ct_ct<BE: ArithBounds>(module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, mut scratch, re1, im1, re2, im2 } = setup(module);

    let ct1 = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));
    let ct2 = encrypt(module, &sk, &re2, &im2, &mut scratch, &mut Source::new([3u8; 32]), &mut Source::new([4u8; 32]));

    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    add_ct_ct(module, &mut ct_res, &ct1, &ct2);
    let (re_out, im_out) = decrypt_decode(module, &ct_res, &sk, &mut scratch);

    // Two independent ciphertexts sum their noise; use 2x tolerance.
    let t = 2.0 * tol(n);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] + re2[j])).abs() < t, "re[{j}]: {} vs {}", re_out[j], re1[j] + re2[j]);
        assert!((im_out[j] - (im1[j] + im2[j])).abs() < t, "im[{j}]: {} vs {}", im_out[j], im1[j] + im2[j]);
    }

    let mut ct_ip = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));
    add_ct_ct_inplace(module, &mut ct_ip, &ct2);
    let (re_ip, im_ip) = decrypt_decode(module, &ct_ip, &sk, &mut scratch);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_pt_ct<BE: ArithBounds>(module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, mut scratch, re1, im1, re2, im2 } = setup(module);

    let ct1 = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));

    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);
    let mut pt2 = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    encode(module, &mut pt2, &re2, &im2, scratch.borrow());

    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    add_pt_ct(module, &mut ct_res, &ct1, &pt2);
    let (re_out, im_out) = decrypt_decode(module, &ct_res, &sk, &mut scratch);

    let t = tol(n);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] + re2[j])).abs() < t, "re[{j}]");
        assert!((im_out[j] - (im1[j] + im2[j])).abs() < t, "im[{j}]");
    }

    let mut ct_ip = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));
    add_pt_ct_inplace(module, &mut ct_ip, &pt2);
    let (re_ip, im_ip) = decrypt_decode(module, &ct_ip, &sk, &mut scratch);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_cleartext_ct<BE: ArithBounds>(module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, mut scratch, re1, im1, .. } = setup(module);

    let c_re: f64 = 0.1;
    let c_im: f64 = -0.05;

    let ct1 = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));

    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    add_cleartext_ct(module, &mut ct_res, &ct1, c_re, c_im);
    let (re_out, im_out) = decrypt_decode(module, &ct_res, &sk, &mut scratch);

    let t = tol(n);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] + c_re)).abs() < t, "re[{j}]: {} vs {}", re_out[j], re1[j] + c_re);
        assert!((im_out[j] - (im1[j] + c_im)).abs() < t, "im[{j}]: {} vs {}", im_out[j], im1[j] + c_im);
    }

    let mut ct_ip = encrypt(module, &sk, &re1, &im1, &mut scratch, &mut Source::new([1u8; 32]), &mut Source::new([2u8; 32]));
    add_cleartext_ct_inplace(module, &mut ct_ip, c_re, c_im);
    let (re_ip, im_ip) = decrypt_decode(module, &ct_ip, &sk, &mut scratch);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + c_re)).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + c_im)).abs() < t, "inplace im[{j}]");
    }
}
