use std::collections::HashMap;

use poulpy_core::{
    GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Dnum, Dsize, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecretPrepared,
        GLWESecretPreparedFactory, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::bin_fhe::{
    bdd_arithmetic::{
        FheUintPrepared, GLWEBlinSelection,
        tests::test_suite::{TEST_FHEUINT_BASE2K, TEST_RANK, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_glwe_blind_selection<BRA: BlindRotationAlgo, BE: Backend>(test_context: &TestContext<BRA, BE>)
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEBlinSelection<u32, BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;

    let base2k: Base2K = TEST_FHEUINT_BASE2K.into();
    let rank: Rank = TEST_RANK.into();
    let k_glwe: TorusPrecision = TorusPrecision(26);
    let k_ggsw: TorusPrecision = TorusPrecision(39);
    let dnum: Dnum = Dnum(3);

    let glwe_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k,
        k: k_glwe,
        rank,
    };
    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: module.n().into(),
        base2k,
        k: k_ggsw,
        rank,
        dnum,
        dsize: Dsize(1),
    };

    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let k: u32 = source.next_u32();

    let mut k_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> =
        FheUintPrepared::<Vec<u8>, u32, BE>::alloc_from_infos(module, &ggsw_infos);
    k_enc_prep.encrypt_sk(
        module,
        k,
        sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let digit = 5;
    let mask: u32 = (1 << digit) - 1;

    // Starting bit
    let mut bit_start: usize = 0;

    let mut data = vec![0i64; 1 << digit];
    data.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    for _ in 0..32_usize.div_ceil(digit) {
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);

        let mut cts_map: HashMap<usize, &mut GLWE<Vec<u8>>> = HashMap::new();
        let mut cts: Vec<GLWE<Vec<u8>>> = Vec::new();

        for value in data.iter().take(1 << digit) {
            pt.encode_coeff_i64(*value, TorusPrecision(base2k.as_u32()), 0);
            let mut ct = GLWE::alloc_from_infos(&glwe_infos);
            ct.encrypt_sk(
                module,
                &pt,
                sk_glwe_prep,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            cts.push(ct);
        }

        for (i, ct) in cts.iter_mut().enumerate() {
            if i.is_multiple_of(3) {
                cts_map.insert(i, ct);
            }
        }

        // How many bits to take
        let bit_size: usize = (32 - bit_start).min(digit);

        module.glwe_blind_selection(
            &mut res,
            cts_map,
            &k_enc_prep,
            bit_start,
            bit_size,
            scratch.borrow(),
        );

        res.decrypt(module, &mut pt, sk_glwe_prep, scratch.borrow());

        let idx = ((k >> bit_start) & mask) as usize;
        if !idx.is_multiple_of(3) {
            assert_eq!(0, pt.decode_coeff_i64(TorusPrecision(base2k.as_u32()), 0));
        } else {
            assert_eq!(
                data[idx],
                pt.decode_coeff_i64(TorusPrecision(base2k.as_u32()), 0)
            );
        }

        bit_start += digit;

        if bit_start >= 32 {
            break;
        }
    }
}
