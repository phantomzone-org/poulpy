use poulpy_core::{
    GGSWEncryptSk, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSW, GGSWLayout, GGSWPreparedFactory, GLWESecret, GLWESecretPrepared,
        GLWESecretPreparedFactory, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};
use rand::RngCore;

use crate::tfhe::bdd_arithmetic::{FheUintPrepared, GGSWBlindRotation};

pub fn test_scalar_to_ggsw_blind_rotation<BE: Backend>()
where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GGSWBlindRotation<u32, BE>
        + GGSWNoise<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: Degree = Degree(1 << 11);
    let base2k: Base2K = Base2K(13);
    let rank: Rank = Rank(1);
    let k_ggsw_res: TorusPrecision = TorusPrecision(39);
    let k_ggsw_apply: TorusPrecision = TorusPrecision(52);

    let ggsw_res_infos: GGSWLayout = GGSWLayout {
        n,
        base2k,
        k: k_ggsw_res,
        rank,
        dnum: Dnum(2),
        dsize: Dsize(1),
    };

    let ggsw_k_infos: GGSWLayout = GGSWLayout {
        n,
        base2k,
        k: k_ggsw_apply,
        rank,
        dnum: Dnum(3),
        dsize: Dsize(1),
    };

    let n_glwe: usize = n.into();

    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);
    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, rank);
    sk_glwe_prep.prepare(&module, &sk_glwe);

    let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_res_infos);

    let mut scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    scalar
        .raw_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as i64);

    let k: u32 = source.next_u32();

    // println!("k: {k}");

    let mut k_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> = FheUintPrepared::<Vec<u8>, u32, BE>::alloc(&module, &ggsw_k_infos);
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

    let max_noise = |col_i: usize| {
        let mut noise: f64 = -(ggsw_res_infos.size() as f64 * base2k.as_usize() as f64) + SIGMA.log2() + 2.0;
        noise += 0.5 * ggsw_res_infos.log_n() as f64;
        if col_i != 0 {
            noise += 0.5 * ggsw_res_infos.log_n() as f64
        }
        noise
    };

    for _ in 0..32_usize.div_ceil(module.log_n()) {
        // By how many bits to left shift
        let mut bit_step: usize = 0;

        for digit in base {
            let mask: u32 = (1 << digit) - 1;

            // How many bits to take
            let bit_size: usize = (32 - bit_start).min(digit);

            module.scalar_to_ggsw_blind_rotation(
                &mut res,
                &scalar,
                &k_enc_prep,
                false,
                bit_start,
                bit_size,
                bit_step,
                scratch.borrow(),
            );

            let rot: i64 = (((k >> bit_start) & mask) << bit_step) as i64;

            let mut scalar_want: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), 1);
            scalar_want.raw_mut().copy_from_slice(scalar.raw());

            module.vec_znx_rotate_inplace(-rot, &mut scalar_want.as_vec_znx_mut(), 0, scratch.borrow());

            // res.print_noise(&module, &sk_glwe_prep, &scalar_want);

            res.assert_noise(&module, &sk_glwe_prep, &scalar_want, &max_noise);

            bit_step += digit;
            bit_start += digit;

            if bit_start >= 32 {
                break;
            }
        }
    }
}
