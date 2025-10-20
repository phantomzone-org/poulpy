use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToMut, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};

use crate::{
    GGLWEExternalProduct, GGLWENoise, GGSWEncryptSk, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWLayout, GGSWPrepare, GGSWPreparedAlloc, GLWESecret, GLWESecretPrepare, GLWESecretPreparedAlloc,
        GLWESwitchingKey, GLWESwitchingKeyLayout,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + VecZnxRotateInplace<BE>
        + GGSWPrepare<BE>
        + GGSWPreparedAlloc<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let k_ggsw: usize = k_in + base2k * di;
                let k_out: usize = k_in; // Better capture noise.

                let n: usize = module.n();
                let dnum: usize = k_in.div_ceil(base2k * di);
                let dsize_in: usize = 1;

                let gglwe_in_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_in.into(),
                    dnum: dnum.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let gglwe_out_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_out.into(),
                    dnum: dnum.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let ggsw_infos: GGSWLayout = GGSWLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ggsw.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank: rank_out.into(),
                };

                let mut ct_gglwe_in: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_in_infos);
                let mut ct_gglwe_out: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_in_infos)
                        | GLWESwitchingKey::external_product_tmp_bytes(module, &gglwe_out_infos, &gglwe_in_infos, &ggsw_infos)
                        | GGSW::encrypt_sk_tmp_bytes(module, &ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk_out_prepared.prepare(module, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                ct_gglwe_in.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                ct_rgsw.encrypt_sk(
                    module,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ct_rgsw);
                ct_rgsw_prepared.prepare(module, &ct_rgsw, scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                ct_gglwe_out.external_product(module, &ct_gglwe_in, &ct_rgsw_prepared, scratch.borrow());

                (0..rank_in).for_each(|i| {
                    module.vec_znx_rotate_inplace(
                        r as i64,
                        &mut sk_in.data.as_vec_znx_mut(),
                        i,
                        scratch.borrow(),
                    ); // * X^{r}
                });

                let var_gct_err_lhs: f64 = SIGMA * SIGMA;
                let var_gct_err_rhs: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = SIGMA * SIGMA;
                let var_a1_err: f64 = 1f64 / 12f64;

                let max_noise: f64 = noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank_out as f64,
                    k_in,
                    k_ggsw,
                );

                ct_gglwe_out
                    .key
                    .assert_noise(module, &sk_out_prepared, &sk_in.data, max_noise + 0.5);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + VecZnxRotateInplace<BE>
        + GGSWPrepare<BE>
        + GGSWPreparedAlloc<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_out: usize = 60;
    let dsize: usize = k_out.div_ceil(base2k);
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let k_ggsw: usize = k_out + base2k * di;

                let n: usize = module.n();
                let dnum: usize = k_out.div_ceil(base2k * di);

                let dsize_in: usize = 1;

                let gglwe_out_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_out.into(),
                    dnum: dnum.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let ggsw_infos: GGSWLayout = GGSWLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ggsw.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank: rank_out.into(),
                };

                let mut ct_gglwe: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_out_infos)
                        | GLWESwitchingKey::external_product_tmp_bytes(module, &gglwe_out_infos, &gglwe_out_infos, &ggsw_infos)
                        | GGSW::encrypt_sk_tmp_bytes(module, &ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk_out_prepared.prepare(module, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                ct_gglwe.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                ct_rgsw.encrypt_sk(
                    module,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ct_rgsw);
                ct_rgsw_prepared.prepare(module, &ct_rgsw, scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                ct_gglwe.external_product_inplace(module, &ct_rgsw_prepared, scratch.borrow());

                (0..rank_in).for_each(|i| {
                    module.vec_znx_rotate_inplace(
                        r as i64,
                        &mut sk_in.data.as_vec_znx_mut(),
                        i,
                        scratch.borrow(),
                    ); // * X^{r}
                });

                let var_gct_err_lhs: f64 = SIGMA * SIGMA;
                let var_gct_err_rhs: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = SIGMA * SIGMA;
                let var_a1_err: f64 = 1f64 / 12f64;

                let max_noise: f64 = noise_ggsw_product(
                    n as f64,
                    base2k * di,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank_out as f64,
                    k_out,
                    k_ggsw,
                );

                ct_gglwe
                    .key
                    .assert_noise(module, &sk_out_prepared, &sk_in.data, max_noise + 0.5);
            }
        }
    }
}
