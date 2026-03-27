use crate::{
    encoding::classical::{decode, encode, encode_tmp_bytes},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::{
        encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
        level::{drop_bits_ct, drop_bits_pt, drop_limbs_ct, drop_limbs_pt},
    },
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, LWEInfos, Rank, TorusPrecision,
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

// Shared bounds alias to avoid repetition across test functions.
pub trait TestBounds: Backend<ScalarPrep = f64, ScalarBig = i64> {}
impl<BE: Backend<ScalarPrep = f64, ScalarBig = i64>> TestBounds for BE {}

// Encode -> encrypt -> apply drop -> decrypt -> decode, then check accuracy.
// Dropping limbs/bits removes LSB precision from the polynomial representation
// but does not change the scaling factor log_delta; tolerance is unchanged.
fn run_drop<BE: TestBounds>(
    module: &Module<BE>,
    drop_ct: impl Fn(&mut CKKSCiphertext<Vec<u8>>),
    drop_pt: impl Fn(&mut CKKSPlaintext<Vec<u8>>),
) where
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
    // Allocate with several extra limbs so drops leave ample precision remaining.
    let k = TorusPrecision(17 * BASE2K);

    let glwe_infos = GLWELayout { n: Degree(n as u32), base2k, k, rank: Rank(1) };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(HW, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    // Allocate the output buffer at full size; the drop closure trims it to
    // match the ciphertext metadata before decryption.
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);

    let scratch_size = encode_tmp_bytes(module)
        .max(encrypt_sk_tmp_bytes(module, &ct))
        .max(decrypt_tmp_bytes(module, &ct));
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

    encode(module, &mut pt, &re_in, &im_in, scratch.borrow());
    encrypt_sk(module, &mut ct, &pt, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    // Apply the same drop to the ciphertext and the output plaintext buffer so
    // that decrypt receives consistent metadata on both sides.
    drop_ct(&mut ct);
    drop_pt(&mut pt_out);

    decrypt(module, &mut pt_out, &ct, &sk_prepared, scratch.borrow());
    let (re_out, im_out) = decode(module, &pt_out);

    // Dropping LSB limbs does not change log_delta, so the tolerance is the
    // same as for a plain encrypt/decrypt round. The residual polynomial
    // precision is still >> log_delta, so quantization error is negligible.
    // Worst-case per-slot error: n * 6 * SIGMA / delta, plus 2x safety margin.
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

/// Drop 2 full limbs (2 * base2k = 104 bits removed from the LSB side).
pub fn test_drop_limbs<BE: TestBounds>(module: &Module<BE>)
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
    const DROP: usize = 2;

    run_drop(module, |ct| drop_limbs_ct(ct, DROP), |pt| drop_limbs_pt(pt, DROP));

    // Verify that metadata was updated correctly.
    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k_init = TorusPrecision(17 * BASE2K);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, 40);
    drop_limbs_ct(&mut ct, DROP);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP as u32 * BASE2K);
    assert_eq!(ct.inner.size(), 17 - DROP);
}

/// Drop 8 bits - a sub-limb drop (8 < base2k = 52), so no full limb is removed.
pub fn test_drop_bits_sublimb<BE: TestBounds>(module: &Module<BE>)
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
    const DROP_BITS: u32 = 8;

    run_drop(module, |ct| drop_bits_ct(ct, DROP_BITS), |pt| drop_bits_pt(pt, DROP_BITS));

    // Verify metadata: k reduced by DROP_BITS, no full limb removed.
    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k_init = TorusPrecision(17 * BASE2K);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, 40);
    drop_bits_ct(&mut ct, DROP_BITS);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP_BITS);
    // DROP_BITS < BASE2K: no full limb should have been removed.
    assert_eq!(ct.inner.size(), 17);
}

/// Drop base2k + 8 bits - spans one full limb plus a sub-limb remainder.
pub fn test_drop_bits_crosslimb<BE: TestBounds>(module: &Module<BE>)
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
    // Drop one full limb (52 bits) plus 8 extra bits.
    const DROP_BITS: u32 = BASE2K + 8;

    run_drop(module, |ct| drop_bits_ct(ct, DROP_BITS), |pt| drop_bits_pt(pt, DROP_BITS));

    // Verify metadata: exactly 1 full limb removed, k reduced by DROP_BITS.
    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k_init = TorusPrecision(17 * BASE2K);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, 40);
    drop_bits_ct(&mut ct, DROP_BITS);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP_BITS);
    assert_eq!(ct.inner.size(), 16);
}
