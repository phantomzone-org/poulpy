use crate::{
    encoding::classical::{decode, encode, encode_tmp_bytes},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared,
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

pub fn test_encrypt_decrypt<BE: Backend<ScalarPrep = f64, ScalarBig = i64>>(module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    const BASE2K: u32 = 52;
    const LOG_DELTA: u32 = 40;
    const HW: usize = 192;

    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);

    let glwe_infos = GLWELayout {
        n: Degree(n as u32),
        base2k,
        k,
        rank: Rank(1),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);

    // Generate and prepare the GLWE secret key.
    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(HW, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    // Allocate plaintext and ciphertext buffers.
    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);

    // Size scratch for the largest of the three operations.
    let scratch_size = encode_tmp_bytes(module)
        .max(encrypt_sk_tmp_bytes(module, &ct))
        .max(decrypt_tmp_bytes(module, &ct));
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

    encode(module, &mut pt, &re_in, &im_in, scratch.borrow());
    encrypt_sk(
        module,
        &mut ct,
        &pt,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    decrypt(module, &mut pt_out, &ct, &sk_prepared, scratch.borrow());
    let (re_out, im_out) = decode(module, &pt_out);

    // Worst-case per-slot error: DFT sums n coefficients each bounded by
    // SIGMA_BOUND = 6 * SIGMA, so |error_j| <= n * 6 * SIGMA / delta.
    // A 2x safety margin is added.
    let tol = 2.0 * (n as f64) * 6.0 * SIGMA / (1u64 << LOG_DELTA) as f64;
    for j in 0..m {
        assert!(
            (re_out[j] - re_in[j]).abs() < tol,
            "re[{j}]: got {}, expected {} (tol={tol})",
            re_out[j],
            re_in[j]
        );
        assert!(
            (im_out[j] - im_in[j]).abs() < tol,
            "im[{j}]: got {}, expected {} (tol={tol})",
            im_out[j],
            im_in[j]
        );
    }
}
