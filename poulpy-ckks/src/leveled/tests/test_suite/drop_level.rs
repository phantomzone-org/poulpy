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

pub trait TestBounds: Backend {}
impl<BE: Backend> TestBounds for BE {}

fn run_drop<BE: TestBounds, CE: Backend<ScalarPrep = f64, ScalarBig = i64>>(
    module: &Module<BE>,
    codec: &Module<CE>,
    drop_ct: impl Fn(&mut CKKSCiphertext<Vec<u8>>),
    drop_pt: impl Fn(&mut CKKSPlaintext<Vec<u8>>),
) where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
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
    assert_eq!(module.n(), codec.n(), "module/codec ring degree mismatch");

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

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(HW, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, LOG_DELTA);

    let mut scratch_codec = ScratchOwned::<CE>::alloc(encode_tmp_bytes(codec));
    let mut scratch = ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(module, &ct).max(decrypt_tmp_bytes(module, &ct)));

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

    encode(codec, &mut pt, &re_in, &im_in, scratch_codec.borrow());
    encrypt_sk(
        module,
        &mut ct,
        &pt,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    drop_ct(&mut ct);
    drop_pt(&mut pt_out);

    decrypt(module, &mut pt_out, &ct, &sk_prepared, scratch.borrow());
    let (re_out, im_out) = decode(codec, &pt_out);

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
pub fn test_drop_limbs<BE: TestBounds, CE: Backend<ScalarPrep = f64, ScalarBig = i64>>(module: &Module<BE>, codec: &Module<CE>)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
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
    const BASE2K: u32 = 52;
    const DROP: usize = 2;

    run_drop(module, codec, |ct| drop_limbs_ct(ct, DROP), |pt| drop_limbs_pt(pt, DROP));

    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k_init = TorusPrecision(17 * BASE2K);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, 40);
    drop_limbs_ct(&mut ct, DROP);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP as u32 * BASE2K);
    assert_eq!(ct.inner.size(), 17 - DROP);
}

/// Drop 8 bits - a sub-limb drop (8 < base2k = 52), so no full limb is removed.
pub fn test_drop_bits_sublimb<BE: TestBounds, CE: Backend<ScalarPrep = f64, ScalarBig = i64>>(
    module: &Module<BE>,
    codec: &Module<CE>,
) where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
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
    const BASE2K: u32 = 52;
    const DROP_BITS: u32 = 8;

    run_drop(
        module,
        codec,
        |ct| drop_bits_ct(ct, DROP_BITS),
        |pt| drop_bits_pt(pt, DROP_BITS),
    );

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
pub fn test_drop_bits_crosslimb<BE: TestBounds, CE: Backend<ScalarPrep = f64, ScalarBig = i64>>(
    module: &Module<BE>,
    codec: &Module<CE>,
) where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
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
    const BASE2K: u32 = 52;
    const DROP_BITS: u32 = BASE2K + 8;

    run_drop(
        module,
        codec,
        |ct| drop_bits_ct(ct, DROP_BITS),
        |pt| drop_bits_pt(pt, DROP_BITS),
    );

    let n = module.n();
    let base2k = Base2K(BASE2K);
    let k_init = TorusPrecision(17 * BASE2K);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, 40);
    drop_bits_ct(&mut ct, DROP_BITS);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP_BITS);
    assert_eq!(ct.inner.size(), 16);
}
