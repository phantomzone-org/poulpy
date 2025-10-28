use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise, SIGMA, ScratchTakeCore,
    layouts::{GGSWLayout, GLWELayout, GLWESecret, GLWESecretPreparedFactory, LWEInfos, LWESecret, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::{
    bdd_arithmetic::{
        BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPreparedFactory, BddKeyPrepared, ExecuteBDDCircuit2WTo1W, FheUint,
        FheUintBlockDebugPrepare, FheUintBlocksPrepare, FheUintBlocksPreparedEncryptSk, FheUintBlocksPreparedFactory,
        FheUintPreparedDebug,
        tests::test_suite::{TEST_BASE2K, TEST_BDD_KEY_LAYOUT, TEST_BLOCK_SIZE, TEST_GGSW_INFOS, TEST_GLWE_INFOS, TEST_N_LWE},
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory},
};

pub fn test_bdd_prepare<BRA: BlindRotationAlgo, BE: Backend>()
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + FheUintBlocksPreparedFactory<u32, BE>
        + FheUintBlocksPreparedEncryptSk<u32, BE>
        + FheUintBlockDebugPrepare<BRA, u32, BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + GGSWNoise<BE>
        + FheUintBlocksPrepare<BRA, u32, BE>
        + ExecuteBDDCircuit2WTo1W<u32, BE>
        + GLWEEncryptSk<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let n_glwe: usize = glwe_infos.n().into();

    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);
    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    // GLWE Secret
    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(&module, &glwe_infos);
    sk_glwe_prep.prepare(&module, &sk_glwe);

    // LWE Secret
    let n_lwe: u32 = TEST_N_LWE;
    let block_size: u32 = TEST_BLOCK_SIZE;
    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size as usize, &mut source_xs);

    // CBT KEY
    let bdd_key_infos: BDDKeyLayout = TEST_BDD_KEY_LAYOUT;

    let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&bdd_key_infos);

    source.fill_bytes(&mut scratch.borrow().data);
    scratch.borrow().data.fill(0);
    bdd_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    let mut bdd_key_prepared: BddKeyPrepared<Vec<u8>, BRA, BE> = BddKeyPrepared::alloc_from_infos(&module, &bdd_key_infos);
    source.fill_bytes(&mut scratch.borrow().data);
    bdd_key_prepared.prepare(&module, &bdd_key, scratch.borrow());

    // GLWE(value)
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_infos);
    let value: u32 = source.next_u32();
    c_enc.encrypt_sk(
        &module,
        value,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    // GGSW(0)
    let mut c_enc_prep_debug: FheUintPreparedDebug<Vec<u8>, u32> =
        FheUintPreparedDebug::<Vec<u8>, u32>::alloc_from_infos(&module, &ggsw_infos);

    // GGSW(value)
    c_enc_prep_debug.prepare(&module, &c_enc, &bdd_key_prepared, scratch.borrow());

    let max_noise = |col_i: usize| {
        let mut noise: f64 = -(ggsw_infos.size() as f64 * TEST_BASE2K as f64) + SIGMA.log2() + 1.0;
        noise += 0.5 * ggsw_infos.log_n() as f64;
        if col_i != 0 {
            noise += 0.5 * ggsw_infos.log_n() as f64
        }
        noise
    };

    // c_enc_prep_debug.print_noise(&module, &sk_glwe_prep, value);

    c_enc_prep_debug.assert_noise(&module, &sk_glwe_prep, value, &max_noise);
}
