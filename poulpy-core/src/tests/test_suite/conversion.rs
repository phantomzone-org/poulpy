use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform},
    layouts::{Backend, FillUniform, Module, Scratch, ScratchOwned, ZnxView},
    source::Source,
};
use rug::Float;

use crate::{
    GLWEDecrypt, GLWEEncryptSk, GLWEFromLWE, GLWENoise, GLWENormalize, GLWEToLWESwitchingKeyEncryptSk, LWEDecrypt, LWEEncryptSk, LWEFromGLWE, LWEToGLWESwitchingKeyEncryptSk, SIGMA, ScratchTakeCore, layouts::{
        Base2K, Degree, Dnum, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWEToLWEKey, GLWEToLWEKeyLayout, GLWEToLWEKeyPrepared, GLWEToLWEKeyPreparedFactory, LWE, LWEInfos, LWELayout, LWEPlaintext, LWESecret, LWEToGLWEKey, LWEToGLWEKeyLayout, LWEToGLWEKeyPrepared, LWEToGLWEKeyPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared
    }
};

pub fn test_glwe_base2k_conversion<BE: Backend>(module: &Module<BE>)
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

    for rank in 1_usize..3 {
        for bases in [[12, 8], [8, 12]] {
            let glwe_infos_in: GLWELayout = GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[0]),
                k: TorusPrecision(34),
                rank: Rank(rank as u32),
            };

            let glwe_infos_out: GLWELayout = GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[1]),
                k: TorusPrecision(34),
                rank: Rank(rank as u32),
            };

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
            sk_prep.prepare(module, &sk);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWE::encrypt_sk_tmp_bytes(module, &glwe_infos_in).max(GLWE::decrypt_tmp_bytes(module, &glwe_infos_out)),
            );

            let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos_in);
            let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos_out);

            let pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos_in);
            let pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos_in);

            ct_in.encrypt_sk(
                module,
                &pt_in,
                &sk_prep,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut data: Vec<Float> = (0..module.n()).map(|_| Float::with_val(128, 0)).collect();
            ct_in.data().decode_vec_float(ct_in.base2k().into(), 0, &mut data);

            ct_out.fill_uniform(ct_out.base2k().into(),&mut source_xa);
            module.glwe_normalize(&mut ct_out, &ct_in, scratch.borrow());
            
            let mut data_conv: Vec<Float> = (0..module.n()).map(|_| Float::with_val(128, 0)).collect();
            ct_out.data().decode_vec_float(ct_out.base2k().into(), 0, &mut data_conv);

            ct_out.assert_noise(module, &sk_prep, &pt_out, -(ct_out.k().as_u32() as f64) + SIGMA.log2() + 0.5);
        }
    }
}

pub fn test_lwe_to_glwe<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEFromLWE<BE>
        + LWEToGLWESwitchingKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + LWEEncryptSk<BE>
        + LWEToGLWEKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let lwe_to_glwe_infos: LWEToGLWEKeyLayout = LWEToGLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(51),
        dnum: Dnum(2),
        rank_out: rank,
    };

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
        rank,
    };

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        LWEToGLWEKey::encrypt_sk_tmp_bytes(module, &lwe_to_glwe_infos)
            | GLWE::from_lwe_tmp_bytes(module, &glwe_infos, &lwe_infos, &lwe_to_glwe_infos)
            | GLWE::decrypt_tmp_bytes(module, &glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk_glwe);
    sk_glwe_prepared.prepare(module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);
    lwe_pt.encode_i64(data, k_lwe_pt);

    let mut lwe_ct: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);
    lwe_ct.encrypt_sk(module, &lwe_pt, &sk_lwe, &mut source_xa, &mut source_xe);

    let mut ksk: LWEToGLWEKey<Vec<u8>> = LWEToGLWEKey::alloc_from_infos(&lwe_to_glwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let mut ksk_prepared: LWEToGLWEKeyPrepared<Vec<u8>, BE> = LWEToGLWEKeyPrepared::alloc_from_infos(module, &ksk);
    ksk_prepared.prepare(module, &ksk, scratch.borrow());

    glwe_ct.from_lwe(module, &lwe_ct, &ksk_prepared, scratch.borrow());

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    glwe_ct.decrypt(module, &mut glwe_pt, &sk_glwe_prepared, scratch.borrow());

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}

pub fn test_glwe_to_lwe<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEFromLWE<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + LWEDecrypt<BE>
        + LWEFromGLWE<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEToLWEKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);

    let glwe_to_lwe_infos: GLWEToLWEKeyLayout = GLWEToLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(51),
        dnum: Dnum(2),
        rank_in: rank,
    };

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
        rank,
    };

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWEToLWEKey::encrypt_sk_tmp_bytes(module, &glwe_to_lwe_infos)
            | LWE::from_glwe_tmp_bytes(module, &lwe_infos, &glwe_infos, &glwe_to_lwe_infos)
            | GLWE::decrypt_tmp_bytes(module, &glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk_glwe);
    sk_glwe_prepared.prepare(module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let a_idx: usize = 1;

    let mut data: Vec<i64> = vec![0i64; module.n()];
    data[a_idx] = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    glwe_pt.encode_vec_i64(&data, k_lwe_pt);

    println!("glwe_pt: {glwe_pt}");

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
    glwe_ct.encrypt_sk(
        module,
        &glwe_pt,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWEKey<Vec<u8>> = GLWEToLWEKey::alloc_from_infos(&glwe_to_lwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);

    let mut ksk_prepared: GLWEToLWEKeyPrepared<Vec<u8>, BE> = GLWEToLWEKeyPrepared::alloc_from_infos(module, &ksk);
    ksk_prepared.prepare(module, &ksk, scratch.borrow());

    lwe_ct.from_glwe(module, &glwe_ct, a_idx, &ksk_prepared, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);
    lwe_ct.decrypt(module, &mut lwe_pt, &sk_lwe);

    assert_eq!(glwe_pt.data.at(0, 0)[a_idx], lwe_pt.data.at(0, 0)[0]);
}
