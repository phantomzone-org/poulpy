use std::time::Instant;

use poulpy_backend::FFT64Ref;
use poulpy_core::{
    TakeGGSW, TakeGLWEPt,
    layouts::{
        GGSWCiphertextLayout, GLWECiphertextLayout, GLWESecret, LWEInfos, LWESecret,
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace,
        SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeSlice, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy, VecZnxDftAddInplace,
        VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume,
        VecZnxIdftApplyTmpA, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, ZnAddNormal, ZnFillUniform,
        ZnNormalizeInplace,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeMatZnxImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl,
    },
    source::Source,
};
use rand::RngCore;

use crate::tfhe::{
    bdd_arithmetic::{
        Add, BDDKey, BDDKeyLayout, BDDKeyPrepared, FheUintBlocks, FheUintBlocksPrep, FheUintBlocksPrepDebug, Sub,
        TEST_BDD_KEY_LAYOUT, TEST_BLOCK_SIZE, TEST_GGSW_INFOS, TEST_GLWE_INFOS, TEST_N_LWE,
    },
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk,
        BlindRotationKeyPrepared, CGGI,
    },
};

#[test]
fn test_bdd_2w_to_1w_fft64_ref() {
    test_bdd_2w_to_1w::<FFT64Ref, CGGI>()
}

fn test_bdd_2w_to_1w<BE: Backend, BRA: BlindRotationAlgo>()
where
    Module<BE>: ModuleNew<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE> + VmpPMatAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Module<BE>: VecZnxAddScalarInplace
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + VmpPrepare<BE>,
    Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGGSW + TakeScalarZnx + TakeSlice,
    Module<BE>: VecZnxCopy + VecZnxNegateInplace + VmpApplyDftToDftTmpBytes + VmpApplyDftToDft<BE> + VmpApplyDftToDftAdd<BE>,
    Module<BE>: VecZnxBigAddInplace<BE> + VecZnxBigAddSmallInplace<BE> + VecZnxBigNormalize<BE>,
    Scratch<BE>: TakeVecZnxDft<BE> + TakeVecZnxBig<BE> + TakeGLWEPt<BE>,
    Module<BE>: VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpA<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigNormalizeTmpBytes
        + SvpPPolAllocBytes
        + VecZnxRotateInplace<BE>
        + VecZnxBigAutomorphismInplace<BE>
        + VecZnxRshInplace<BE>
        + VecZnxDftCopy<BE>
        + VecZnxAutomorphismInplace<BE>
        + VecZnxBigSubSmallNegateInplace<BE>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<BE>
        + VecZnxRotate
        + ZnFillUniform
        + ZnAddNormal
        + ZnNormalizeInplace<BE>,
    BE: Backend
        + ScratchOwnedAllocImpl<BE>
        + ScratchOwnedBorrowImpl<BE>
        + TakeVecZnxDftImpl<BE>
        + ScratchAvailableImpl<BE>
        + TakeVecZnxImpl<BE>
        + TakeScalarZnxImpl<BE>
        + TakeSvpPPolImpl<BE>
        + TakeVecZnxBigImpl<BE>
        + TakeVecZnxDftSliceImpl<BE>
        + TakeMatZnxImpl<BE>
        + TakeVecZnxSliceImpl<BE>,
    BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<BE, BlindRotationKeyPrepared<Vec<u8>, BRA, BE>>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, BE>: BlincRotationExecute<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<BE>,
{
    let glwe_infos: GLWECiphertextLayout = TEST_GLWE_INFOS;
    let ggsw_infos: GGSWCiphertextLayout = TEST_GGSW_INFOS;

    let n_glwe: usize = glwe_infos.n().into();

    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);
    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = sk_glwe.prepare_alloc(&module, scratch.borrow());

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    println!("a: {a}");
    println!("b: {b}");

    let mut a_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);
    let mut b_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);
    let mut c_enc: FheUintBlocks<Vec<u8>, u32> = FheUintBlocks::<Vec<u8>, u32>::alloc(&module, &glwe_infos);
    let mut c_enc_prep_debug: FheUintBlocksPrepDebug<Vec<u8>, u32> =
        FheUintBlocksPrepDebug::<Vec<u8>, u32>::alloc(&module, &ggsw_infos);
    let mut c_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);

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

    let now: Instant = Instant::now();
    let bdd_key: BDDKey<Vec<u8>, Vec<u8>, BRA> = BDDKey::encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &bdd_key_infos,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    let bdd_key_prepared: BDDKeyPrepared<Vec<u8>, Vec<u8>, BRA, BE> = bdd_key.prepare_alloc(&module, scratch.borrow());
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
