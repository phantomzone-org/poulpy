use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToMut, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};

use crate::{
    GGSWEncryptSk, GGSWExternalProduct, GGSWNoise, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWLayout, GGSWPrepare, GGSWPreparedAlloc, GLWESecret, GLWESecretPrepare, GLWESecretPreparedAlloc,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWExternalProduct<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + GGSWPrepare<BE>
        + GGSWPreparedAlloc<BE>
        + VecZnxRotateInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = k_in + base2k * di;

            let k_out: usize = k_in; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(base2k * di);
            let dnum_in: usize = k_in.div_euclid(base2k * di);
            let dsize_in: usize = 1;

            let ggsw_in_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_in: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_in_infos);
            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);
            let mut pt_in: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_apply: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_in.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_apply.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_apply_infos)
                    | GGSW::encrypt_sk_tmp_bytes(module, &ggsw_in_infos)
                    | GGSW::external_product_tmp_bytes(module, &ggsw_out_infos, &ggsw_in_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            ggsw_apply.encrypt_sk(
                module,
                &pt_apply,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ggsw_in.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ct_rhs_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ggsw_apply);
            ct_rhs_prepared.prepare(module, &ggsw_apply, scratch.borrow());

            ggsw_out.external_product(module, &ggsw_in, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_in.as_vec_znx_mut(), 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    0.5,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank as f64,
                    k_in,
                    k_apply,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_prepared, &pt_in, max_noise);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWExternalProduct<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + GGSWPrepare<BE>
        + GGSWPreparedAlloc<BE>
        + VecZnxRotateInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_out: usize = 60;
    let dsize: usize = k_out.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = k_out + base2k * di;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(di * base2k);
            let dnum_in: usize = k_out.div_euclid(base2k * di);
            let dsize_in: usize = 1;

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);

            let mut pt_in: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_apply: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_in.fill_ternary_prob(0, 0.5, &mut source_xs);

            let k: usize = 1;

            pt_apply.to_mut().raw_mut()[k] = 1; //X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_apply_infos)
                    | GGSW::encrypt_sk_tmp_bytes(module, &ggsw_out_infos)
                    | GGSW::external_product_tmp_bytes(module, &ggsw_out_infos, &ggsw_out_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            ggsw_apply.encrypt_sk(
                module,
                &pt_apply,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ggsw_out.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ct_rhs_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ggsw_apply);
            ct_rhs_prepared.prepare(module, &ggsw_apply, scratch.borrow());

            ggsw_out.external_product_inplace(module, &ct_rhs_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_in.as_vec_znx_mut(), 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise = |_col_j: usize| -> f64 {
                noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    0.5,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank as f64,
                    k_out,
                    k_apply,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_prepared, &pt_in, max_noise);
        }
    }
}
