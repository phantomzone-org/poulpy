use itertools::Itertools;
use poulpy_backend::FFT64Ref;
use poulpy_core::layouts::{
    Digits, GGSWCiphertext, GGSWCiphertextLayout, GLWECiphertext, GLWECiphertextLayout, GLWESecret,
    prepared::{GGSWCiphertextPrepared, PrepareAlloc},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::arithmetic::{ADD_OP32, SLL_OP32, SRA_OP32, SRL_OP32, SUB_OP32};

fn test_int_ops<B: Backend>()
where
    Module<B>: ModuleNew<B> + SvpPPolAlloc<B> + SvpPrepare<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let logn: usize = 8_usize;
    let base2k: usize = 15_usize;
    let k_glwe: usize = base2k * 2;
    let k_ggsw: usize = base2k * 3;
    let rank: usize = 1_usize;
    let rows: usize = k_glwe.div_ceil(base2k);

    let module: Module<B> = Module::<B>::new(1 << logn);
    let mut source: Source = Source::new([0u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch = ScratchOwned::alloc(1 << 20);

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
