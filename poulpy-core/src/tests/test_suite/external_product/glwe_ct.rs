use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxRotateInplace},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};

use crate::{
    GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct, GLWENoise, GLWENormalize, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + VecZnxRotateInplace<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_in: usize = 17;
    let base2k_key: usize = 13;
    let base2k_out: usize = 15;
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(base2k_key);
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ggsw: usize = k_in + base2k_key * dsize;
            let k_out: usize = k_ggsw; // Better capture noise

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(k_ggsw * dsize);

            let glwe_in_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_in.into(),
                k: k_in.into(),
                rank: rank.into(),
            };

            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ggsw.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);
            let mut glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(base2k_in, &mut pt_in.data, 0, &mut source_xa);

            pt_in.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_apply_infos)
                    | GLWE::encrypt_sk_tmp_bytes(module, &glwe_in_infos)
                    | GLWE::external_product_tmp_bytes(module, &glwe_out_infos, &glwe_in_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            ggsw_apply.encrypt_sk(
                module,
                &pt_ggsw,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            glwe_in.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ct_ggsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ggsw_apply);
            ct_ggsw_prepared.prepare(module, &ggsw_apply, scratch.borrow());

            glwe_out.external_product(module, &glwe_in, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_in.data, 0, scratch.borrow());

            module.glwe_normalize(&mut pt_out, &pt_in, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                base2k_key * max_dsize,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_in,
                k_ggsw,
            );

            glwe_out.assert_noise(module, &sk_prepared, &pt_out, max_noise + 0.5);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + VecZnxRotateInplace<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_out: usize = 17;
    let base2k_key: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(base2k_key);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ggsw: usize = k_out + base2k_key * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(base2k_out * max_dsize);

            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ggsw.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(base2k_out, &mut pt_want.data, 0, &mut source_xa);

            pt_want.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_apply_infos)
                    | GLWE::encrypt_sk_tmp_bytes(module, &glwe_out_infos)
                    | GLWE::external_product_tmp_bytes(module, &glwe_out_infos, &glwe_out_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            ggsw_apply.encrypt_sk(
                module,
                &pt_ggsw,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            glwe_out.encrypt_sk(
                module,
                &pt_want,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ct_ggsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &ggsw_apply);
            ct_ggsw_prepared.prepare(module, &ggsw_apply, scratch.borrow());

            glwe_out.external_product_inplace(module, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_inplace(k as i64, &mut pt_want.data, 0, scratch.borrow());

            let var_gct_err_lhs: f64 = SIGMA * SIGMA;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = SIGMA * SIGMA;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                base2k_key * max_dsize,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_out,
                k_ggsw,
            );

            glwe_out.assert_noise(module, &sk_prepared, &pt_want, max_noise + 0.5);
        }
    }
}
