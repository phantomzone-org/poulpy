use dashu_float::{FBig, round::mode::HalfEven};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxNormalize},
    layouts::{DeviceBuf, FillUniform, Module, Scratch, ScratchOwned, ZnxView},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEFromLWE, GLWENoise, GLWENormalize,
    GLWEToLWESwitchingKeyEncryptSk, LWEDecrypt, LWEEncryptSk, LWEFromGLWE, LWEToGLWESwitchingKeyEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWEToLWEKey,
        GLWEToLWEKeyLayout, GLWEToLWEKeyPrepared, GLWEToLWEKeyPreparedFactory, LWE, LWEInfos, LWELayout, LWEPlaintext, LWESecret,
        LWEToGLWEKey, LWEToGLWEKeyLayout, LWEToGLWEKeyPrepared, LWEToGLWEKeyPreparedFactory, Rank, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};

pub fn test_glwe_base2k_conversion<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let base2k: usize = params.base2k;

    for rank in 1_usize..3 {
        for bases in [[base2k, base2k - 3], [base2k - 3, base2k]] {
            let k_in = 4 * bases[0] + 1;
            let k_out = 4 * bases[0] + 1;

            let glwe_infos_in = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[0] as u32),
                k: TorusPrecision(k_in as u32),
                rank: Rank(rank as u32),
            })
            .unwrap();

            let glwe_infos_out: GLWELayout = GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[1] as u32),
                k: TorusPrecision(k_out as u32),
                rank: Rank(rank as u32),
            };

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prep: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prep, &sk);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .glwe_encrypt_sk_tmp_bytes(&glwe_infos_in)
                    .max(module.glwe_noise_tmp_bytes(&glwe_infos_out)),
            );

            let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos_in);
            let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos_out);

            let pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos_in);
            let pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos_out);

            module.glwe_encrypt_sk(
                &mut ct_in,
                &pt_in,
                &sk_prep,
                &glwe_infos_in,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut data: Vec<FBig<HalfEven>> = (0..module.n()).map(|_| FBig::ZERO).collect();
            ct_in.data().decode_vec_float(ct_in.base2k().into(), 0, &mut data);

            ct_out.fill_uniform(ct_out.base2k().into(), &mut source_xa);
            module.glwe_normalize(&mut ct_out, &ct_in, scratch.borrow());

            let mut data_conv: Vec<FBig<HalfEven>> = (0..module.n()).map(|_| FBig::ZERO).collect();
            ct_out.data().decode_vec_float(ct_out.base2k().into(), 0, &mut data_conv);

            let noise_have = module.glwe_noise(&ct_out, &pt_out, &sk_prep, scratch.borrow()).std().log2();
            let noise_max = -(k_out as f64) + DEFAULT_SIGMA_XE.log2() + 0.50;

            assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}")
        }
    }
}

pub fn test_lwe_to_glwe<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEFromLWE<BE>
        + LWEToGLWESwitchingKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + LWEEncryptSk<BE>
        + LWEToGLWEKeyPreparedFactory<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);
    let base2k: usize = params.base2k;

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);
    let k_ksk = 5 * base2k + 1;
    let k_glwe = 4 * base2k + 1;
    let k_lwe = 4 * base2k + 1;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let lwe_to_glwe_infos = EncryptionLayout::new_from_default_sigma(LWEToGLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        dnum: Dnum(2),
        rank_out: rank,
    })
    .unwrap();

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32 - 1),
        k: TorusPrecision(k_glwe as u32),
        rank,
    };

    let lwe_infos = EncryptionLayout::new_from_default_sigma(LWELayout {
        n: n_lwe,
        base2k: Base2K(base2k as u32 - 2),
        k: TorusPrecision(k_lwe as u32),
    })
    .unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).lwe_to_glwe_key_encrypt_sk_tmp_bytes(&lwe_to_glwe_infos)
            | (module).glwe_from_lwe_tmp_bytes(&glwe_infos, &lwe_infos, &lwe_to_glwe_infos)
            | (module).glwe_decrypt_tmp_bytes(&glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);
    lwe_pt.encode_i64(data, k_lwe_pt);

    let mut lwe_ct: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut lwe_ct,
        &lwe_pt,
        &sk_lwe,
        &lwe_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut ksk: LWEToGLWEKey<Vec<u8>> = LWEToGLWEKey::alloc_from_infos(&lwe_to_glwe_infos);

    module.lwe_to_glwe_key_encrypt_sk(
        &mut ksk,
        &sk_lwe,
        &sk_glwe_prepared,
        &lwe_to_glwe_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let mut ksk_prepared: LWEToGLWEKeyPrepared<DeviceBuf<BE>, BE> = module.lwe_to_glwe_key_prepared_alloc_from_infos(&ksk);
    module.lwe_to_glwe_key_prepare(&mut ksk_prepared, &ksk, scratch.borrow());

    module.glwe_from_lwe(&mut glwe_ct, &lwe_ct, &ksk_prepared, scratch.borrow());

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    module.glwe_decrypt(&glwe_ct, &mut glwe_pt, &sk_glwe_prepared, scratch.borrow());

    let mut lwe_pt_conv = LWEPlaintext::alloc(glwe_pt.base2k(), lwe_pt.max_k());

    module.vec_znx_normalize(
        lwe_pt_conv.data_mut(),
        glwe_pt.base2k().as_usize(),
        0,
        0,
        lwe_pt.data(),
        lwe_pt.base2k().as_usize(),
        0,
        scratch.borrow(),
    );

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt_conv.data.at(0, 0)[0]);
}

pub fn test_glwe_to_lwe<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEFromLWE<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + LWEDecrypt<BE>
        + LWEFromGLWE<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEToLWEKeyPreparedFactory<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);
    let base2k: usize = params.base2k;
    let k_ksk = 5 * base2k + 1;
    let k_glwe = 4 * base2k + 1;
    let k_lwe = 4 * base2k + 1;

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);

    let glwe_to_lwe_infos = EncryptionLayout::new_from_default_sigma(GLWEToLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        dnum: Dnum(2),
        rank_in: rank,
    })
    .unwrap();

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32 - 1),
        k: TorusPrecision(k_glwe as u32),
        rank,
    })
    .unwrap();

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe,
        base2k: Base2K(base2k as u32 - 2),
        k: TorusPrecision(k_lwe as u32),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).glwe_to_lwe_key_encrypt_sk_tmp_bytes(&glwe_to_lwe_infos)
            | (module).lwe_from_glwe_tmp_bytes(&lwe_infos, &glwe_infos, &glwe_to_lwe_infos)
            | (module).glwe_decrypt_tmp_bytes(&glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let a_idx: usize = 1;

    let mut data: Vec<i64> = vec![0i64; module.n()];
    data[a_idx] = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    glwe_pt.encode_vec_i64(&data, k_lwe_pt);

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
    module.glwe_encrypt_sk(
        &mut glwe_ct,
        &glwe_pt,
        &sk_glwe_prepared,
        &glwe_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWEKey<Vec<u8>> = GLWEToLWEKey::alloc_from_infos(&glwe_to_lwe_infos);

    module.glwe_to_lwe_key_encrypt_sk(
        &mut ksk,
        &sk_lwe,
        &sk_glwe,
        &glwe_to_lwe_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut lwe_ct: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);

    let mut ksk_prepared: GLWEToLWEKeyPrepared<DeviceBuf<BE>, BE> = module.glwe_to_lwe_key_prepared_alloc_from_infos(&ksk);
    module.glwe_to_lwe_key_prepare(&mut ksk_prepared, &ksk, scratch.borrow());

    module.lwe_from_glwe(&mut lwe_ct, &glwe_ct, a_idx, &ksk_prepared, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);
    module.lwe_decrypt(&lwe_ct, &mut lwe_pt, &sk_lwe, scratch.borrow());

    let mut glwe_pt_conv = GLWEPlaintext::<Vec<u8>, ()>::alloc(glwe_ct.n(), lwe_pt.base2k(), lwe_pt.max_k());

    module.vec_znx_normalize(
        glwe_pt_conv.data_mut(),
        lwe_pt.base2k().as_usize(),
        0,
        0,
        glwe_pt.data(),
        glwe_ct.base2k().as_usize(),
        0,
        scratch.borrow(),
    );

    assert_eq!(glwe_pt_conv.data.at(0, 0)[a_idx], lwe_pt.data.at(0, 0)[0]);
}
