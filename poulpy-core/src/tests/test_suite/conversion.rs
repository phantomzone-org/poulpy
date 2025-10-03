use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScratchOwned, ZnxView},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, GLWECiphertext, GLWECiphertextLayout, GLWEPlaintext, GLWESecret, GLWEToLWEKey, GLWEToLWEKeyLayout,
    LWECiphertext, LWECiphertextLayout, LWEPlaintext, LWESecret, LWEToGLWESwitchingKey, LWEToGLWESwitchingKeyLayout, Rank, Rows,
    TorusPrecision,
    prepared::{GLWESecretPrepared, GLWEToLWESwitchingKeyPrepared, LWEToGLWESwitchingKeyPrepared, PrepareAlloc},
};

pub fn test_lwe_to_glwe<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxAutomorphismInplace<B>
        + ZnNormalizeInplace<B>
        + ZnFillUniform
        + ZnAddNormal,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
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
        rows: Rows(2),
        rank_out: rank,
    };

    let glwe_infos: GLWECiphertextLayout = GLWECiphertextLayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
        rank,
    };

    let lwe_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWEToGLWESwitchingKey::encrypt_sk_scratch_space(module, &lwe_to_glwe_infos)
            | GLWECiphertext::from_lwe_scratch_space(module, &glwe_infos, &lwe_infos, &lwe_to_glwe_infos)
            | GLWECiphertext::decrypt_scratch_space(module, &glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(&lwe_infos);
    lwe_pt.encode_i64(data, k_lwe_pt);

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&lwe_infos);
    lwe_ct.encrypt_sk(module, &lwe_pt, &sk_lwe, &mut source_xa, &mut source_xe);

    let mut ksk: LWEToGLWESwitchingKey<Vec<u8>> = LWEToGLWESwitchingKey::alloc(&lwe_to_glwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut glwe_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_infos);

    let ksk_prepared: LWEToGLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    glwe_ct.from_lwe(module, &lwe_ct, &ksk_prepared, scratch.borrow());

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_infos);
    glwe_ct.decrypt(module, &mut glwe_pt, &sk_glwe_prepared, scratch.borrow());

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}

pub fn test_glwe_to_lwe<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub
        + SvpPrepare<B>
        + SvpPPolAllocBytes
        + SvpPPolAlloc<B>
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddScalarInplace
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxAutomorphismInplace<B>
        + ZnNormalizeInplace<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeSvpPPolImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);

    let glwe_to_lwe_infos: GLWEToLWEKeyLayout = GLWEToLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(51),
        rows: Rows(2),
        rank_in: rank,
    };

    let glwe_infos: GLWECiphertextLayout = GLWECiphertextLayout {
        n: n_glwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
        rank,
    };

    let lwe_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe,
        base2k: Base2K(17),
        k: TorusPrecision(34),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWEToLWEKey::encrypt_sk_scratch_space(module, &glwe_to_lwe_infos)
            | LWECiphertext::from_glwe_scratch_space(module, &lwe_infos, &glwe_infos, &glwe_to_lwe_infos)
            | GLWECiphertext::decrypt_scratch_space(module, &glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_infos);
    glwe_pt.encode_coeff_i64(data, k_lwe_pt, 0);

    let mut glwe_ct = GLWECiphertext::alloc(&glwe_infos);
    glwe_ct.encrypt_sk(
        module,
        &glwe_pt,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: GLWEToLWEKey<Vec<u8>> = GLWEToLWEKey::alloc(&glwe_to_lwe_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&lwe_infos);

    let ksk_prepared: GLWEToLWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    lwe_ct.from_glwe(module, &glwe_ct, &ksk_prepared, scratch.borrow());

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(&lwe_infos);
    lwe_ct.decrypt(module, &mut lwe_pt, &sk_lwe);

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt.data.at(0, 0)[0]);
}
