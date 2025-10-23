use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise, ScratchTakeCore,
    layouts::{GGSWLayout, GLWELayout, GLWESecret, GLWESecretPreparedFactory, LWEInfos, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::{
    bdd_arithmetic::{
        BDDKeyEncryptSk, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUintBlockDebugPrepare, FheUintBlocks,
        FheUintBlocksPrepare, FheUintBlocksPrepared, FheUintBlocksPreparedEncryptSk, FheUintBlocksPreparedFactory, Slt,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS},
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory},
};

pub fn test_bdd_slt<BRA: BlindRotationAlgo, BE: Backend>()
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

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(&module, &glwe_infos);
    sk_glwe_prep.prepare(&module, &sk_glwe);

    let mut res: FheUintBlocks<Vec<u8>, u32> = FheUintBlocks::<Vec<u8>, u32>::alloc_from_infos(&module, &glwe_infos);
    let mut a_enc_prep: FheUintBlocksPrepared<Vec<u8>, u32, BE> =
        FheUintBlocksPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);
    let mut b_enc_prep: FheUintBlocksPrepared<Vec<u8>, u32, BE> =
        FheUintBlocksPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    source.fill_bytes(&mut scratch.borrow().data);
    a_enc_prep.encrypt_sk(
        &module,
        a,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    source.fill_bytes(&mut scratch.borrow().data);
    b_enc_prep.encrypt_sk(
        &module,
        b,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    // d + a
    res.slt(&module, &a_enc_prep, &b_enc_prep, scratch.borrow());

    assert_eq!(
        res.decrypt(&module, &sk_glwe_prep, scratch.borrow()),
        ((a as i32) < (b as i32)) as u32
    );
}
