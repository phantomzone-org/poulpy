use poulpy_core::{
    GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWESecretPrepared, GLWESecretPreparedFactory, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::bdd_arithmetic::{FheUintPrepared, GLWEBlindRotation};

pub fn test_glwe_to_glwe_blind_rotation<BE: Backend>()
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEBlindRotation<u32, BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: Degree = Degree(1 << 11);
    let base2k: Base2K = Base2K(13);
    let rank: Rank = Rank(1);
    let k_glwe: TorusPrecision = TorusPrecision(26);
    let k_ggsw: TorusPrecision = TorusPrecision(39);
    let dnum: Dnum = Dnum(3);

    let glwe_infos: GLWELayout = GLWELayout {
        n,
        base2k,
        k: k_glwe,
        rank,
    };
    let ggsw_infos: GGSWLayout = GGSWLayout {
        n,
        base2k,
        k: k_ggsw,
        rank,
        dnum,
        dsize: Dsize(1),
    };

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

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let mut test_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    let mut data: Vec<i64> = vec![0i64; module.n()];
    data.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
    test_glwe.encode_vec_i64(&data, base2k.as_usize().into());

    let k: u32 = source.next_u32();

    let mut k_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> = FheUintPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_infos);
    k_enc_prep.encrypt_sk(
        &module,
        k,
        &sk_glwe_prep,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let base: [usize; 2] = [6, 5];

    assert_eq!(base.iter().sum::<usize>(), module.log_n());

    // Starting bit
    let mut bit_start: usize = 0;

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);

    for _ in 0..32_usize.div_ceil(module.log_n()) {
        // By how many bits to left shift
        let mut bit_step: usize = 0;

        for digit in base {
            let mask: u32 = (1 << digit) - 1;

            // How many bits to take
            let bit_size: usize = (32 - bit_start).min(digit);

            module.glwe_to_glwe_blind_rotation(
                &mut res,
                &test_glwe,
                &k_enc_prep,
                false,
                bit_start,
                bit_size,
                bit_step,
                scratch.borrow(),
            );

            res.decrypt(&module, &mut pt, &sk_glwe_prep, scratch.borrow());

            assert_eq!(
                (((k >> bit_start) & mask) << bit_step) as i64,
                pt.decode_coeff_i64(base2k.as_usize().into(), 0)
            );

            bit_step += digit;
            bit_start += digit;

            if bit_start >= 32 {
                break;
            }
        }
    }
}
