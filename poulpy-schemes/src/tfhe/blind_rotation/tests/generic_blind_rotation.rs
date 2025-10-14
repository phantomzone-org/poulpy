use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf,
        SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAdd, VecZnxDftAddInplace,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftSubInplace, VecZnxDftZero, VecZnxFillUniform, VecZnxIdftApply,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpBytes, VecZnxMulXpMinusOneInplace, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, ZnAddNormal,
        ZnFillUniform, ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScratchOwned, ZnxView},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeSliceImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl,
        TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::tfhe::blind_rotation::{
    BlincRotationExecute, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk, BlindRotationKeyLayout,
    BlindRotationKeyPrepared, CGGI, LookUpTable, cggi_blind_rotate_scratch_space, mod_switch_2n,
};

use poulpy_core::layouts::{
    GLWE, GLWELayout, GLWEPlaintext, GLWESecret, LWE, LWECiphertextLayout, LWECiphertextToRef, LWEInfos, LWEPlaintext, LWESecret,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};

pub fn test_blind_rotation<B>(module: &Module<B>, n_lwe: usize, block_size: usize, extension_factor: usize)
where
    Module<B>: VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + SvpPPolBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApply<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + VecZnxDftApply<B>
        + VecZnxDftZero<B>
        + SvpApplyDftToDft<B>
        + VecZnxDftSubInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace<B>
        + SvpPrepare<B>
        + SvpPPolAlloc<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxAddScalarInplace
        + VecZnxRotateInplace<B>
        + VecZnxSwitchRing
        + VecZnxSub
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + ZnFillUniform
        + ZnAddNormal
        + VecZnxRotateInplaceTmpBytes
        + ZnNormalizeInplace<B>,
    B: Backend
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeVecZnxDftSliceImpl<B>
        + ScratchAvailableImpl<B>
        + TakeVecZnxImpl<B>
        + TakeVecZnxSliceImpl<B>
        + TakeSliceImpl<B>,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 19;
    let k_lwe: usize = 24;
    let k_brk: usize = 3 * base2k;
    let rows_brk: usize = 2; // Ensures first limb is noise-free.
    let k_lut: usize = base2k;
    let k_res: usize = 2 * base2k;
    let rank: usize = 1;

    let log_message_modulus: usize = 4;

    let message_modulus: usize = 1 << log_message_modulus;

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let brk_infos: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: n_glwe.into(),
        n_lwe: n_lwe.into(),
        base2k: base2k.into(),
        k: k_brk.into(),
        dnum: rows_brk.into(),
        rank: rank.into(),
    };

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_res.into(),
        rank: rank.into(),
    };

    let lwe_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe.into(),
        k: k_lwe.into(),
        base2k: base2k.into(),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(BlindRotationKey::generate_from_sk_scratch_space(
        module, &brk_infos,
    ));

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch_br: ScratchOwned<B> = ScratchOwned::<B>::alloc(cggi_blind_rotate_scratch_space(
        module,
        block_size,
        extension_factor,
        &glwe_infos,
        &brk_infos,
    ));

    let mut brk: BlindRotationKey<Vec<u8>, CGGI> = BlindRotationKey::<Vec<u8>, CGGI>::alloc(&brk_infos);

    brk.encrypt_sk(
        module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_infos);

    let x: i64 = 15 % (message_modulus as i64);

    pt_lwe.encode_i64(x, (log_message_modulus + 1).into());

    lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let f = |x: i64| -> i64 { 2 * x + 1 };

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = f(i as i64));

    let mut lut: LookUpTable = LookUpTable::alloc(module, base2k, k_lut, extension_factor);
    lut.set(module, &f_vec, log_message_modulus + 1);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

    let brk_prepared: BlindRotationKeyPrepared<Vec<u8>, CGGI, B> = brk.prepare_alloc(module, scratch.borrow());

    brk_prepared.execute(module, &mut res, &lwe, &lut, scratch_br.borrow());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);

    res.decrypt(module, &mut pt_have, &sk_glwe_dft, scratch.borrow());

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space

    mod_switch_2n(
        2 * lut.domain_size(),
        &mut lwe_2n,
        &lwe.to_ref(),
        lut.rotation_direction(),
    );

    let pt_want: i64 = (lwe_2n[0]
        + lwe_2n[1..]
            .iter()
            .zip(sk_lwe.raw())
            .map(|(x, y)| x * y)
            .sum::<i64>())
        & (2 * lut.domain_size() - 1) as i64;

    lut.rotate(module, pt_want);

    // First limb should be exactly equal (test are parameterized such that the noise does not reach
    // the first limb)
    assert_eq!(pt_have.data.at(0, 0), lut.data[0].at(0, 0));

    // Verify that it effectively compute f(x)
    let mut have: i64 = pt_have.decode_coeff_i64((log_message_modulus + 1).into(), 0);

    // Get positive representative and assert equality
    have = (have + message_modulus as i64) % (message_modulus as i64);

    assert_eq!(have, f(x) % (message_modulus as i64));
}
