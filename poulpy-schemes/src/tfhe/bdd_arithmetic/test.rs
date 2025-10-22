use std::time::Instant;

use poulpy_backend::FFT64Ref;
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWENoise, ScratchTakeCore,
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
        Add, BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W,
        FheUintBlockDebugPrepare, FheUintBlocks, FheUintBlocksPrepare, FheUintBlocksPrepared, FheUintBlocksPreparedDebug,
        FheUintBlocksPreparedEncryptSk, FheUintBlocksPreparedFactory, Sub, TEST_BDD_KEY_LAYOUT, TEST_BLOCK_SIZE, TEST_GGSW_INFOS,
        TEST_GLWE_INFOS, TEST_N_LWE,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, CGGI},
};

#[test]
fn test_bdd_2w_to_1w_fft64_ref() {
    test_bdd_2w_to_1w::<FFT64Ref, CGGI>()
}

fn test_bdd_2w_to_1w<BE: Backend, BRA: BlindRotationAlgo>()
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
        + ExecuteBDDCircuit2WTo1W<u32, BE>,
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

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(&module, &glwe_infos);
    sk_glwe_prep.prepare(&module, &sk_glwe);

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    println!("a: {a}");
    println!("b: {b}");

    let mut a_enc_prep: FheUintBlocksPrepared<Vec<u8>, u32, BE> =
        FheUintBlocksPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);
    let mut b_enc_prep: FheUintBlocksPrepared<Vec<u8>, u32, BE> =
        FheUintBlocksPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);
    let mut c_enc: FheUintBlocks<Vec<u8>, u32> = FheUintBlocks::<Vec<u8>, u32>::alloc(&module, &glwe_infos);
    let mut c_enc_prep_debug: FheUintBlocksPreparedDebug<Vec<u8>, u32> =
        FheUintBlocksPreparedDebug::<Vec<u8>, u32>::alloc(&module, &ggsw_infos);
    let mut c_enc_prep: FheUintBlocksPrepared<Vec<u8>, u32, BE> =
        FheUintBlocksPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);

    a_enc_prep.encrypt_sk(
        &module,
        a,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    b_enc_prep.encrypt_sk(
        &module,
        b,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let start: Instant = Instant::now();
    c_enc.sub(&module, &a_enc_prep, &b_enc_prep, scratch.borrow());

    let duration: std::time::Duration = start.elapsed();
    println!("add: {} ms", duration.as_millis());

    println!(
        "have: {}",
        c_enc.decrypt(&module, &sk_glwe_prep, scratch.borrow())
    );
    println!("want: {}", a.wrapping_sub(b));
    println!(
        "noise: {:?}",
        c_enc.noise(&module, &sk_glwe_prep, a.wrapping_sub(b), scratch.borrow())
    );

    let n_lwe: u32 = TEST_N_LWE;
    let block_size: u32 = TEST_BLOCK_SIZE;

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size as usize, &mut source_xs);

    let bdd_key_infos: BDDKeyLayout = TEST_BDD_KEY_LAYOUT;

    let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&bdd_key_infos);

    let now: Instant = Instant::now();
    bdd_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    let mut bdd_key_prepared: BDDKeyPrepared<Vec<u8>, BRA, BE> = BDDKeyPrepared::alloc_from_infos(&module, &bdd_key_infos);
    bdd_key_prepared.prepare(&module, &bdd_key, scratch.borrow());
    println!("BDD-KGEN: {} ms", now.elapsed().as_millis());

    let now: Instant = Instant::now();
    c_enc_prep_debug.prepare(&module, &c_enc, &bdd_key_prepared, scratch.borrow());
    println!("CBT: {} ms", now.elapsed().as_millis());

    c_enc_prep_debug.noise(&module, &sk_glwe_prep, a.wrapping_sub(b));

    let now: Instant = Instant::now();
    c_enc_prep.prepare(&module, &c_enc, &bdd_key_prepared, scratch.borrow());
    println!("CBT: {} ms", now.elapsed().as_millis());

    let start: Instant = Instant::now();
    c_enc.add(&module, &c_enc_prep, &b_enc_prep, scratch.borrow());

    let duration: std::time::Duration = start.elapsed();
    println!("add: {} ms", duration.as_millis());

    println!(
        "have: {}",
        c_enc.decrypt(&module, &sk_glwe_prep, scratch.borrow())
    );
    println!("want: {}", b.wrapping_add(a.wrapping_sub(b)));
    println!(
        "noise: {:?}",
        c_enc.noise(
            &module,
            &sk_glwe_prep,
            b.wrapping_add(a.wrapping_sub(b)),
            scratch.borrow()
        )
    );
}
