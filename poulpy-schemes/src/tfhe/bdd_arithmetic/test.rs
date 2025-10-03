use std::time::Instant;

use poulpy_backend::FFT64Avx;
use poulpy_core::{
    TakeGGSW, TakeGLWEPt,
    layouts::{
        Digits, GGLWEAutomorphismKeyLayout, GGLWETensorKeyLayout, GGSWCiphertextLayout, GLWECiphertextLayout, GLWESecret,
        GLWEToLWEKeyLayout, LWESecret,
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace,
        SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy, VecZnxDftAddInplace, VecZnxDftAlloc,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace,
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
    bdd_arithmetic::{Add, BDDKey, BDDKeyLayout, BDDKeyPrepared, FheUintBlocks, FheUintBlocksPrep, FheUintBlocksPrepDebug, Sub},
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk,
        BlindRotationKeyLayout, BlindRotationKeyPrepared, CGGI,
    },
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};

#[test]
fn test_int_ops_fft64_avx() {
    test_int_ops::<FFT64Avx, CGGI>()
}

fn test_int_ops<BE: Backend, BRA: BlindRotationAlgo>()
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
    Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGGSW + TakeScalarZnx,
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
    let n_glwe: usize = 1024;
    let base2k: usize = 13_usize;
    let k_glwe: usize = base2k * 2;
    let k_ggsw: usize = base2k * 3;
    let rank: usize = 2_usize;
    let rows: usize = k_glwe.div_ceil(base2k);

    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);
    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(module.n().into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = sk_glwe.prepare_alloc(&module, scratch.borrow());

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    let glwe_infos: GLWECiphertextLayout = GLWECiphertextLayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: k_ggsw.into(),
        rank: rank.into(),
    };

    let ggsw_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: k_ggsw.into(),
        rows: rows.into(),
        digits: Digits(1),
        rank: rank.into(),
    };

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

    let n_lwe: usize = 574;
    let block_size: usize = 7;

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let lwe_ks_infos: GLWEToLWEKeyLayout = GLWEToLWEKeyLayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: k_ggsw.into(),
        rows: rows.into(),
        rank_in: rank.into(),
    };

    let k_brk: usize = 4 * base2k;
    let rows_brk: usize = 3;

    let k_atk: usize = 4 * base2k;
    let rows_atk: usize = 3;

    let k_tsk: usize = 4 * base2k;
    let rows_tsk: usize = 3;

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            rows: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GGLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_atk.into(),
            rows: rows_atk.into(),
            rank: rank.into(),
            digits: Digits(1),
        },
        layout_tsk: GGLWETensorKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            rows: rows_tsk.into(),
            digits: Digits(1),
            rank: rank.into(),
        },
    };

    let bdd_key_infos: BDDKeyLayout = BDDKeyLayout {
        cbt: cbt_infos,
        ks: lwe_ks_infos,
    };

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

    // c_enc_prep_debug.noise(&module, &sk_glwe_prep, a.wrapping_sub(b));

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
