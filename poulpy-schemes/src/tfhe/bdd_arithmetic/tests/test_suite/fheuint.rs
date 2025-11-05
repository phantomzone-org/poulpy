use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWERotate, GLWESub, GLWETrace,
    layouts::{GLWELayout, GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::tfhe::{
    bdd_arithmetic::{
        BDDKeyPrepared, FheUint, ScratchTakeBDD,
        tests::test_suite::{TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_fhe_uint_sext<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<Vec<u8>, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);

    for j in 0..3 {
        for i in 0..32 {
            let a: u32 = 0xFFFFFFFF >> i;

            a_enc.encrypt_sk(
                module,
                a,
                sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            a_enc.sext(module, j, keys, scratch.borrow());

            // println!("{:08x} -> {:08x} {:08x}", a, sext(a, j), a_enc.decrypt(module, sk, scratch.borrow()));

            assert_eq!(sext(a, j), a_enc.decrypt(module, sk, scratch.borrow()));
        }
    }
}

pub fn sext(x: u32, byte: usize) -> u32 {
    x | ((x >> (byte << 3)) & 1) * (0xFFFF_FFFF & (0xFFFF_FFFF << (byte << 3)))
}

pub fn test_fhe_uint_splice_u8<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<Vec<u8>, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    b_enc.encrypt_sk(
        module,
        b,
        sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    a_enc.encrypt_sk(
        module,
        a,
        sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    for dst in 0..4 {
        for src in 0..4 {
            c_enc.splice_u8(module, dst, src, &a_enc, &b_enc, keys, scratch.borrow());

            let rj: u32 = (dst << 3) as u32;
            let ri: u32 = (src << 3) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);

            let c_want: u32 = ((a_r & 0xFFFF_FF00) | (b_r & 0x0000_00FF)).rotate_left(rj);

            assert_eq!(c_want, c_enc.decrypt(module, sk, scratch.borrow()));
        }
    }
}

pub fn test_fhe_uint_splice_u16<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<Vec<u8>, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    b_enc.encrypt_sk(
        module,
        b,
        sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    a_enc.encrypt_sk(
        module,
        a,
        sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    for dst in 0..2 {
        for src in 0..2 {
            c_enc.splice_u16(module, dst, src, &a_enc, &b_enc, keys, scratch.borrow());
            let rj: u32 = (dst << 4) as u32;
            let ri: u32 = (src << 4) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);
            let c_want: u32 = ((a_r & 0xFFFF_0000) | (b_r & 0x0000_FFFF)).rotate_left(rj);
            assert_eq!(c_want, c_enc.decrypt(module, sk, scratch.borrow()));
        }
    }
}
