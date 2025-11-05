use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise, ScratchTakeCore,
    layouts::{GGSWLayout, GLWELayout, GLWESecretPreparedFactory, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::{
    bdd_arithmetic::{
        BDDKeyEncryptSk, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare,
        FheUintPrepareDebug, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory, Or,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory},
};

pub fn test_bdd_or<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + FheUintPreparedFactory<u32, BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPrepareDebug<BRA, u32, BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + GGSWNoise<BE>
        + FheUintPrepare<BRA, u32, BE>
        + ExecuteBDDCircuit2WTo1W<u32, BE>
        + GLWEEncryptSk<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;
    let bdd_key_prepared: &BDDKeyPrepared<Vec<u8>, BRA, BE> = &test_context.bdd_key;

    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut res: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
    let mut a_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> =
        FheUintPrepared::<Vec<u8>, u32, BE>::alloc_from_infos(module, &ggsw_infos);
    let mut b_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> =
        FheUintPrepared::<Vec<u8>, u32, BE>::alloc_from_infos(module, &ggsw_infos);

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    source.fill_bytes(&mut scratch.borrow().data);
    a_enc_prep.encrypt_sk(
        module,
        a,
        sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    source.fill_bytes(&mut scratch.borrow().data);
    b_enc_prep.encrypt_sk(
        module,
        b,
        sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    res.or(
        module,
        &a_enc_prep,
        &b_enc_prep,
        bdd_key_prepared,
        scratch.borrow(),
    );

    assert_eq!(res.decrypt(module, sk_glwe_prep, scratch.borrow()), a | b);
}
