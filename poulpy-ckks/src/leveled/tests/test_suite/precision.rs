use super::CKKSTestParams;
use crate::{
    encoding::classical::{decode, encode},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

fn max_slot_error(re_out: &[f64], im_out: &[f64], re_in: &[f64], im_in: &[f64]) -> f64 {
    let mut max_err: f64 = 0.0;
    for j in 0..re_in.len() {
        max_err = max_err.max((re_out[j] - re_in[j]).abs());
        max_err = max_err.max((im_out[j] - im_in[j]).abs());
    }
    max_err
}

pub fn test_precision<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
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
    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

    for log_delta in [30, 40, 50] {
        let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, log_delta);
        encode(&mut pt, &re_in, &im_in);

        // encode → decode
        let (re_out, im_out) = decode(&pt);
        let enc_dec_bits = -(max_slot_error(&re_out, &im_out, &re_in, &im_in).log2());

        // encode → encrypt → decrypt → decode
        let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, log_delta);
        let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, log_delta);
        let mut scratch = ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(module, &ct).max(decrypt_tmp_bytes(module, &ct)));
        let mut source_xa = Source::new([1u8; 32]);
        let mut source_xe = Source::new([2u8; 32]);
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
        let (re_out2, im_out2) = decode(&pt_out);
        let full_bits = -(max_slot_error(&re_out2, &im_out2, &re_in, &im_in).log2());

        eprintln!(
            "precision: n={n} base2k={} log_delta={log_delta}: \
             encode_decode={enc_dec_bits:.1} bits  full_pipeline={full_bits:.1} bits",
            params.base2k,
        );
    }
}
