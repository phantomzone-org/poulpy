use poulpy_hal::{
    api::{
        DFT, IDFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApply, SvpApplyInplace, SvpPPolAlloc,
        SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAdd,
        VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftSubABInplace, VecZnxDftZero, VecZnxFillUniform, VecZnxIDFTTmpBytes,
        VecZnxMulXpMinusOneInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRotateInplace, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
        VmpPMatAlloc, VmpPrepare, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScratchOwned, ZnxView},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl,
        TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::tfhe::blind_rotation::{
    BlincRotationExecute, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk, BlindRotationKeyPrepared, CGGI,
    LookUpTable, cggi_blind_rotate_scratch_space, mod_switch_2n,
};

use poulpy_core::layouts::{
    GLWECiphertext, GLWEPlaintext, GLWESecret, Infos, LWECiphertext, LWECiphertextToRef, LWEPlaintext, LWESecret,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};

pub fn test_blind_rotation<B>(module: &Module<B>, n_lwe: usize, block_size: usize, extension_factor: usize)
where
    Module<B>: VecZnxBigAllocBytes
        + VecZnxDftAllocBytes
        + SvpPPolAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIDFTTmpBytes
        + IDFT<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + DFT<B>
        + VecZnxDftZero<B>
        + SvpApply<B>
        + VecZnxDftSubABInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubABInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace
        + SvpPrepare<B>
        + SvpPPolAlloc<B>
        + SvpApplyInplace<B>
        + IDFTConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxAddScalarInplace
        + VecZnxRotateInplace
        + VecZnxSwithcDegree
        + VecZnxSub
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VmpApply<B>
        + VmpApplyAdd<B>
        + ZnFillUniform
        + ZnAddNormal
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
        + TakeVecZnxSliceImpl<B>,
{
    let n: usize = module.n();
    let basek: usize = 19;
    let k_lwe: usize = 24;
    let k_brk: usize = 3 * basek;
    let rows_brk: usize = 2; // Ensures first limb is noise-free.
    let k_lut: usize = basek;
    let k_res: usize = 2 * basek;
    let rank: usize = 1;

    let log_message_modulus = 4;

    let message_modulus: usize = 1 << log_message_modulus;

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(BlindRotationKey::generate_from_sk_scratch_space(
        module, basek, k_brk, rank,
    ));

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: GLWESecretPrepared<Vec<u8>, B> = sk_glwe.prepare_alloc(module, scratch.borrow());

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch_br: ScratchOwned<B> = ScratchOwned::<B>::alloc(cggi_blind_rotate_scratch_space(
        module,
        block_size,
        extension_factor,
        basek,
        k_res,
        k_brk,
        rows_brk,
        rank,
    ));

    let mut brk: BlindRotationKey<Vec<u8>, CGGI> =
        BlindRotationKey::<Vec<u8>, CGGI>::alloc(n, n_lwe, basek, k_brk, rows_brk, rank);

    brk.encrypt_sk(
        module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe);

    let x: i64 = 15 % (message_modulus as i64);

    pt_lwe.encode_i64(x, log_message_modulus + 1);

    lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let f = |x: i64| -> i64 { 2 * x + 1 };

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = f(i as i64));

    let mut lut: LookUpTable = LookUpTable::alloc(module, basek, k_lut, extension_factor);
    lut.set(module, &f_vec, log_message_modulus + 1);

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_res, rank);

    let brk_prepared: BlindRotationKeyPrepared<Vec<u8>, CGGI, B> = brk.prepare_alloc(module, scratch.borrow());

    brk_prepared.execute(module, &mut res, &lwe, &lut, scratch_br.borrow());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_res);

    res.decrypt(module, &mut pt_have, &sk_glwe_dft, scratch.borrow());

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space

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
    let mut have: i64 = pt_have.decode_coeff_i64(log_message_modulus + 1, 0);

    // Get positive representative and assert equality
    have = (have + message_modulus as i64) % (message_modulus as i64);

    assert_eq!(have, f(x) % (message_modulus as i64));
}
