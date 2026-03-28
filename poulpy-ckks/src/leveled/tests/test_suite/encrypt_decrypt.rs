use super::CKKSTestParams;
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

pub fn test_encrypt_decrypt<BE: Backend, CE: Backend<ScalarPrep = f64, ScalarBig = i64>>(
    module: &Module<BE>,
    codec: &Module<CE>,
    params: CKKSTestParams,
)
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
    assert_eq!(module.n(), codec.n(), "module/codec ring degree mismatch");

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
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);

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
    decrypt(module, &mut pt_out, &ct, &sk_prepared, scratch.borrow());
    let (re_out, im_out) = decode(codec, &pt_out);

    let tol = 2.0 * (n as f64) * 6.0 * SIGMA / (1u64 << params.log_delta) as f64;
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
