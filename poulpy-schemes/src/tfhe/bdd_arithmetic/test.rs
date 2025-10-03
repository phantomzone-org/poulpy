use std::time::Instant;

use poulpy_backend::FFT64Avx;
use poulpy_core::{
    layouts::{prepared::{GLWEToLWESwitchingKeyPrepared, PrepareAlloc}, Digits, GGLWELayoutInfos, GGSWCiphertextLayout, GLWECiphertextLayout, GLWESecret, GLWEToLWESwitchingKey, GLWEToLWESwitchingKeyLayout}, TakeGGSW, TakeGLWEPt
};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPrepare, TakeScalarZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::bdd_arithmetic::{ADD_OP32, FheUintBlocks, FheUintBlocksPrep, SLL_OP32, SRA_OP32, SRL_OP32, SUB_OP32};

#[test]
fn test_int_ops_fft64_avx(){
    test_int_ops::<FFT64Avx>()
}

fn test_int_ops<BE: Backend>()
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
{
    let logn: usize = 10_usize;
    let base2k: usize = 15_usize;
    let k_glwe: usize = base2k * 2;
    let k_ggsw: usize = base2k * 3;
    let rank: usize = 1_usize;
    let rows: usize = k_glwe.div_ceil(base2k);

    let module: Module<BE> = Module::<BE>::new(1 << logn);
    let mut source: Source = Source::new([0u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(module.n().into(), rank.into());
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_prep = sk.prepare_alloc(&module, scratch.borrow());

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

    let lwe_ks_infos: GLWEToLWESwitchingKeyLayout = GLWEToLWESwitchingKeyLayout{
        n: module.n().into(),
        base2k: base2k.into(),
        k: k_ggsw.into(),
        rows: rows.into(),
        rank_in: rank.into(),
    };

    let lwe_ks: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc(&lwe_ks_infos);

    let mut lwe_ks_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, BE> = GLWEToLWESwitchingKeyPrepared::alloc(&module, &lwe_ks_infos);

    let mut a_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);
    let mut b_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);
    let mut c_enc: FheUintBlocks<Vec<u8>, u32> = FheUintBlocks::<Vec<u8>, u32>::alloc(&module, &glwe_infos);
    let mut c_enc_prep: FheUintBlocksPrep<Vec<u8>, BE, u32> = FheUintBlocksPrep::<Vec<u8>, BE, u32>::alloc(&module, &ggsw_infos);

    a_enc_prep.encrypt_sk(
        &module,
        a,
        &sk_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    b_enc_prep.encrypt_sk(
        &module,
        b,
        &sk_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let start: Instant = Instant::now();
    c_enc.add(&module, &a_enc_prep, &b_enc_prep, scratch.borrow());

    
    let duration: std::time::Duration = start.elapsed();
    println!(
        "add: {} ms",
        duration.as_millis()
    );

    println!("have: {}", c_enc.decrypt(&module, &sk_prep, scratch.borrow()));
    println!("want: {}", a.wrapping_add(b));

    //c_enc_prep.prepare(module, c_enc, lwe_ks, key, scratch);

    // macro_rules! eval_test {
    // ($c: ident, $a:ident, $b:ident, $want: expr) => {
    // let mut outputs = (0..$op_type.word_size())
    // .map(|_| GLWECiphertext::alloc(&glwe_infos))
    // .collect_vec();
    //
    // $op_type.execute(&module, &mut outputs, &$a, &$b, scratch.borrow());
    //
    // let have = outputs
    // .iter()
    // .map(|b| decrypt_bit(&module, b, &sk_prep, scratch.borrow()))
    // .collect_vec();
    //
    // assert_eq!(have, $want);
    // };
    // }

    // eval_test!(
    // a_enc_prep,
    // b_enc_prep,
    // a.wrapping_add(b)
    // );
    // eval_test!(
    // SUB_OP32,
    // a_enc_prep,
    // b_enc_prep,
    // a.wrapping_sub(b)
    // );
    //
    // eval_test!(
    // SLL_OP32,
    // shamt_enc_prep,
    // a_enc_prep,
    // a.wrapping_shl(shamt)
    // );
    //
    // eval_test!(
    // SRL_OP32,
    // shamt_enc_prep,
    // a_enc_prep,
    // a.wrapping_shr(shamt)
    // );
    //
    // eval_test!(
    // SRA_OP32,
    // shamt_enc_prep,
    // a_enc_prep,
    // ((a as i32).wrapping_shr(shamt)) as u32
    // );
}
