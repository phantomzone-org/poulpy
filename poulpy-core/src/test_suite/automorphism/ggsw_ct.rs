use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismInplace},
    layouts::{Module, ScalarZnx, ScalarZnxAsVecZnxBackendMut, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEToGGSWKeyEncryptSk, GGSWAutomorphism, GGSWEncryptSk, GGSWNoise, GLWEAutomorphismKeyEncryptSk,
    ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPreparedFactory, GGSW, GGSWInfos, GGSWLayout, GGSWToBackendMut,
        GLWEAutomorphismKey, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGLWEToGGSWKeyPrepared, GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_keyswitch,
};

pub fn test_ggsw_automorphism<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssign<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum_in: usize = k_in / in_base2k;
            let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

            let dsize_in: usize = 1;

            let ggsw_in_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ggsw_out_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let tsk_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct_in: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_in_layout);
            let mut ct_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_layout);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_layout);
            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ct_in)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk)
                    | module.ggsw_automorphism_tmp_bytes(&ct_out, &ct_in, &auto_key, &tsk),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct_out);
            sk.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p,
                &sk,
                &auto_key_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk,
                &tsk_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            module.ggsw_encrypt_sk(
                &mut ct_in,
                &pt_scalar,
                &sk_prepared,
                &ggsw_in_layout,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_layout);
            module.glwe_automorphism_key_prepare(&mut auto_key_prepared, &auto_key, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            {
                let mut ct_out_backend = <GGSW<Vec<u8>> as GGSWToBackendMut<BE>>::to_backend_mut(&mut ct_out);
                let ct_in_backend = <GGSW<Vec<u8>> as crate::layouts::GGSWToBackendRef<BE>>::to_backend_ref(&ct_in);
                module.ggsw_automorphism(
                    &mut ct_out_backend,
                    &ct_in_backend,
                    &auto_key_prepared,
                    &tsk_prepared,
                    &mut scratch.borrow(),
                );
            }

            module.vec_znx_automorphism_inplace(
                p,
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxAsVecZnxBackendMut<BE>>::as_vec_znx_backend_mut(&mut pt_scalar),
                0,
                &mut scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    key_base2k * dsize,
                    col_j,
                    var_xs,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank as f64,
                    k_in,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            for row in 0..ct_out.dnum().as_usize() {
                for col in 0..ct_out.rank().as_usize() + 1 {
                    let noise = ct_out
                        .noise(module, row, col, &pt_scalar.to_ref(), &sk_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    let max_noise = max_noise(col);
                    assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_automorphism_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssign<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p: i64 = -1;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let dnum_in: usize = k_out / out_base2k;
            let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);
            let dsize_in: usize = 1;

            let ggsw_out_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let tsk_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_layout);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_layout);
            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ct)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk)
                    | module.ggsw_automorphism_tmp_bytes(&ct, &ct, &auto_key, &tsk),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct);
            sk.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p,
                &sk,
                &auto_key_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk,
                &tsk_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            module.ggsw_encrypt_sk(
                &mut ct,
                &pt_scalar,
                &sk_prepared,
                &ggsw_out_layout,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_layout);
            module.glwe_automorphism_key_prepare(&mut auto_key_prepared, &auto_key, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            {
                let mut ct_backend = <GGSW<Vec<u8>> as GGSWToBackendMut<BE>>::to_backend_mut(&mut ct);
                module.ggsw_automorphism_inplace(&mut ct_backend, &auto_key_prepared, &tsk_prepared, &mut scratch.borrow());
            }

            module.vec_znx_automorphism_inplace(
                p,
                &mut <ScalarZnx<Vec<u8>> as ScalarZnxAsVecZnxBackendMut<BE>>::as_vec_znx_backend_mut(&mut pt_scalar),
                0,
                &mut scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    key_base2k * dsize,
                    col_j,
                    var_xs,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank as f64,
                    k_out,
                    k_ksk,
                    k_tsk,
                ) + 4.0
            };

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    let noise_have: f64 = ct
                        .noise(module, row, col, &pt_scalar.to_ref(), &sk_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    let noise_max: f64 = max_noise(col);
                    assert!(noise_have <= noise_max, "noise_have:{noise_have} > noise_max:{noise_max}")
                }
            }
        }
    }
}
