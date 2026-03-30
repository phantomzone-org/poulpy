use super::CKKSTestParams;
use crate::{
    encoding::classical::{decode, encode},
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
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

fn run_drop<BE: Backend>(
    module: &Module<BE>,
    params: CKKSTestParams,
    drop_ct: impl Fn(&mut CKKSCiphertext<Vec<u8>>),
    drop_pt: impl Fn(&mut CKKSPlaintext<Vec<u8>>),
) where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);

    let mut scratch = ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(module, &ct).max(decrypt_tmp_bytes(module, &ct)));

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

    encode(&mut pt, &re_in, &im_in);
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
    let (re_out, im_out) = decode(&pt_out);

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

pub fn test_drop_limbs<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    const DROP: usize = 2;

    run_drop(module, params, |ct| drop_limbs_ct(ct, DROP), |pt| drop_limbs_pt(pt, DROP));

    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k_init = TorusPrecision(params.k);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, params.log_delta);
    drop_limbs_ct(&mut ct, DROP);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP as u32 * params.base2k);
    assert_eq!(ct.inner.size(), params.k.div_ceil(params.base2k) as usize - DROP);
}

pub fn test_drop_bits_sublimb<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    const DROP_BITS: u32 = 8;

    run_drop(
        module,
        params,
        |ct| drop_bits_ct(ct, DROP_BITS),
        |pt| drop_bits_pt(pt, DROP_BITS),
    );

    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k_init = TorusPrecision(params.k);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, params.log_delta);
    drop_bits_ct(&mut ct, DROP_BITS);
    assert_eq!(ct.inner.k().0, k_init.0 - DROP_BITS);
    assert_eq!(ct.inner.size(), params.k.div_ceil(params.base2k) as usize);
}

pub fn test_drop_bits_crosslimb<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let drop_bits = params.base2k + 8;

    run_drop(
        module,
        params,
        |ct| drop_bits_ct(ct, drop_bits),
        |pt| drop_bits_pt(pt, drop_bits),
    );

    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k_init = TorusPrecision(params.k);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k_init, params.log_delta);
    drop_bits_ct(&mut ct, drop_bits);
    assert_eq!(ct.inner.k().0, k_init.0 - drop_bits);
    assert_eq!(ct.inner.size(), (params.k - drop_bits).div_ceil(params.base2k) as usize);
}
