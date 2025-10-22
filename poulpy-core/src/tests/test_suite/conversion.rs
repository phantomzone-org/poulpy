use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned, ZnxView},
    source::Source,
};

use crate::{
    GLWEDecrypt, GLWEEncryptSk, GLWEFromLWE, GLWEToLWESwitchingKeyEncryptSk, LWEDecrypt, LWEEncryptSk,
    LWEToGLWESwitchingKeyEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWEToLWEKeyLayout,
        GLWEToLWESwitchingKey, GLWEToLWESwitchingKeyPreparedFactory, LWE, LWELayout, LWEPlaintext, LWESecret,
        LWEToGLWESwitchingKey, LWEToGLWESwitchingKeyLayout, LWEToGLWESwitchingKeyPreparedFactory, Rank, TorusPrecision,
        prepared::{GLWESecretPrepared, GLWEToLWESwitchingKeyPrepared, LWEToGLWESwitchingKeyPrepared},
    },
};

pub fn test_lwe_to_glwe<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEFromLWE<BE>
        + LWEToGLWESwitchingKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + LWEEncryptSk<BE>
        + LWEToGLWESwitchingKeyPreparedFactory<BE>,
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

    let lwe_to_glwe_infos: LWEToGLWESwitchingKeyLayout = LWEToGLWESwitchingKeyLayout {
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
        LWEToGLWESwitchingKey::encrypt_sk_tmp_bytes(module, &lwe_to_glwe_infos)
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

    let mut ksk: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc_from_infos(&lwe_to_glwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let mut ksk_prepared: LWEToGLWESwitchingKeyPrepared<Vec<u8>, BE> =
        LWEToGLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
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
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEToLWESwitchingKeyPreparedFactory<BE>,
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
        GLWEToLWESwitchingKey::encrypt_sk_tmp_bytes(module, &glwe_to_lwe_infos)
            | LWE::from_glwe_tmp_bytes(module, &lwe_infos, &glwe_infos, &glwe_to_lwe_infos)
            | GLWE::decrypt_tmp_bytes(module, &glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk_glwe);
    sk_glwe_prepared.prepare(module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);
    glwe_pt.encode_coeff_i64(data, k_lwe_pt, 0);

    let mut glwe_ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
    glwe_ct.encrypt_sk(
        module,
        &glwe_pt,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWESwitchingKey<Vec<u8>> = GLWEToLWESwitchingKey::alloc_from_infos(&glwe_to_lwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);

    let mut ksk_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, BE> =
        GLWEToLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
    ksk_prepared.prepare(module, &ksk, scratch.borrow());

    lwe_ct.from_glwe(module, &glwe_ct, &ksk_prepared, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);
    lwe_ct.decrypt(module, &mut lwe_pt, &sk_lwe);

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}
