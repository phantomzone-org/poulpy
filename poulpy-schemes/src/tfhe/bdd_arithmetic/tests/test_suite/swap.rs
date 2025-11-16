use itertools::Itertools;
use poulpy_core::{
    GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk,
    layouts::{GGSW, GGSWPrepared, GGSWPreparedFactory, GLWELayout, GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::{
    bdd_arithmetic::{
        Cswap, FheUint, FheUintPrepared, GLWEBlindRetrieval, GLWEBlindRetriever, ScratchTakeBDD,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_fhe_uint_swap<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + Cswap<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut s: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);
    let mut s_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ggsw_infos);

    let a: u32 = source_xa.next_u32();
    let b: u32 = source_xa.next_u32();

    for bit in [0, 1] {
        let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
        let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);

        a_enc.encrypt_sk(
            module,
            a,
            sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        b_enc.encrypt_sk(
            module,
            b,
            sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut pt: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), 1);
        pt.raw_mut()[0] = bit;
        s.encrypt_sk(
            module,
            &pt,
            sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );
        s_prepared.prepare(module, &s, scratch.borrow());

        module.cswap(&mut a_enc, &mut b_enc, &s_prepared, scratch.borrow());

        let (a_want, b_want) = if bit == 0 { (a, b) } else { (b, a) };

        assert_eq!(a_want, a_enc.decrypt(module, sk, scratch.borrow()));
        assert_eq!(b_want, b_enc.decrypt(module, sk, scratch.borrow()));
    }
}

pub fn test_glwe_blind_retrieval_statefull<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWEBlindRetrieval<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let data: Vec<u32> = (0..25).map(|i| i as u32).collect_vec();

    let mut data_enc: Vec<FheUint<Vec<u8>, u32>> = (0..data.len())
        .map(|i| {
            let mut ct: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
            ct.encrypt_sk(
                module,
                data[i],
                sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            ct
        })
        .collect_vec();

    for idx in 0..data.len() as u32 {
        let mut idx_enc = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
        idx_enc.encrypt_sk(
            module,
            idx,
            sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        module.glwe_blind_retrieval_statefull(&mut data_enc, &idx_enc, 0, 5, scratch.borrow());

        assert_eq!(
            data[idx as usize],
            data_enc[0].decrypt(module, sk, scratch.borrow())
        );

        module.glwe_blind_retrieval_statefull_rev(&mut data_enc, &idx_enc, 0, 5, scratch.borrow());

        for i in 0..data.len() {
            assert_eq!(data[i], data_enc[i].decrypt(module, sk, scratch.borrow()))
        }
    }
}

pub fn test_glwe_blind_retriever<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWEBlindRetrieval<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeBDD<u32, BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let data: Vec<u32> = (0..25).map(|i| i as u32).collect_vec();

    let data_enc: Vec<FheUint<Vec<u8>, u32>> = (0..data.len())
        .map(|i| {
            let mut ct: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(&glwe_infos);
            ct.encrypt_sk(
                module,
                data[i],
                sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            ct
        })
        .collect_vec();

    let mut retriever: GLWEBlindRetriever = GLWEBlindRetriever::alloc(&glwe_infos, data.len());
    for idx in 0..data.len() as u32 {
        let offset = 2;
        let mut idx_enc: FheUintPrepared<Vec<u8>, u32, BE> = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
        idx_enc.encrypt_sk(
            module,
            idx << offset,
            sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut res: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&glwe_infos);
        retriever.retrieve(
            module,
            &mut res,
            &data_enc,
            &idx_enc,
            offset,
            scratch.borrow(),
        );

        assert_eq!(
            data[idx as usize],
            res.decrypt(module, sk, scratch.borrow())
        );
    }
}
