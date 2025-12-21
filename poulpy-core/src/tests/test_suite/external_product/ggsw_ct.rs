use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToMut, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};

use crate::{
    GGSWEncryptSk, GGSWExternalProduct, GGSWNoise, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWInfos, GGSWLayout, GGSWPreparedFactory, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWExternalProduct<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxRotateInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let in_base2k: usize = 17;
    let key_base2k: usize = 13;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_apply: usize = k_in + key_base2k * dsize;

            let k_out: usize = k_in; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(key_base2k * dsize);
            let dnum_in: usize = k_in / in_base2k;
            let dsize_in: usize = 1;

            let ggsw_in_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_apply.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
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

            ggsw_in.encrypt_sk(module, &pt_in, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

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
                    key_base2k * dsize,
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

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    assert!(
                        ggsw_out
                            .noise(module, row, col, &pt_in, &sk_prepared, scratch.borrow())
                            .std()
                            .log2()
                            <= max_noise(col)
                    )
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_external_product_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWExternalProduct<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxRotateInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let out_base2k: usize = 17;
    let key_base2k: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_apply: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(dsize * key_base2k);
            let dnum_in: usize = k_out / out_base2k;
            let dsize_in: usize = 1;

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_apply.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
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

            ggsw_out.encrypt_sk(module, &pt_in, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

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
                    key_base2k * dsize,
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

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    assert!(
                        ggsw_out
                            .noise(module, row, col, &pt_in, &sk_prepared, scratch.borrow())
                            .std()
                            .log2()
                            <= max_noise(col)
                    )
                }
            }
        }
    }
}
