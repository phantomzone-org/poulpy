use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace,
        VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    GLWEOperations, GLWEPacker,
    layouts::{
        GGLWEAutomorphismKey, GGLWEAutomorphismKeyLayout, GLWECiphertext, GLWECiphertextLayout, GLWEPlaintext, GLWESecret,
        prepared::{GGLWEAutomorphismKeyPrepared, GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_glwe_packing<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxAutomorphism
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxNegateInplace
        + VecZnxRshInplace<B>
        + VecZnxRotateInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + VecZnxRotate
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxFillUniform
        + VecZnxSubABInplace
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
        + VecZnxCopy,
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
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let n: usize = module.n();
    let base2k: usize = 18;
    let k_ct: usize = 36;
    let pt_k: usize = 18;
    let rank: usize = 3;
    let digits: usize = 1;
    let k_ksk: usize = k_ct + base2k * digits;

    let rows: usize = k_ct.div_ceil(base2k * digits);

    let glwe_out_infos: GLWECiphertextLayout = GLWECiphertextLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_ct.into(),
        rank: rank.into(),
    };

    let key_infos: GGLWEAutomorphismKeyLayout = GGLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_ksk.into(),
        rank: rank.into(),
        digits: digits.into(),
        rows: rows.into(),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, &glwe_out_infos)
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, &key_infos)
            | GLWEPacker::scratch_space(module, &glwe_out_infos, &key_infos),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&glwe_out_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    pt.encode_vec_i64(&data, pt_k.into());

    let gal_els: Vec<i64> = GLWEPacker::galois_elements(module);

    let mut auto_keys: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> = HashMap::new();
    let mut tmp: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(&key_infos);
    gal_els.iter().for_each(|gal_el| {
        tmp.encrypt_sk(
            module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );
        let atk_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, B> = tmp.prepare_alloc(module, scratch.borrow());
        auto_keys.insert(*gal_el, atk_prepared);
    });

    let log_batch: usize = 0;

    let mut packer: GLWEPacker = GLWEPacker::new(&glwe_out_infos, log_batch);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_out_infos);

    ct.encrypt_sk(
        module,
        &pt,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let log_n: usize = module.log_n();

    (0..n >> log_batch).for_each(|i| {
        ct.encrypt_sk(
            module,
            &pt,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        pt.rotate_inplace(module, -(1 << log_batch), scratch.borrow()); // X^-batch * pt

        if reverse_bits_msb(i, log_n as u32).is_multiple_of(5) {
            packer.add(module, Some(&ct), &auto_keys, scratch.borrow());
        } else {
            packer.add(
                module,
                None::<&GLWECiphertext<Vec<u8>>>,
                &auto_keys,
                scratch.borrow(),
            )
        }
    });

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_out_infos);
    packer.flush(module, &mut res);

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        if i.is_multiple_of(5) {
            *x = reverse_bits_msb(i, log_n as u32) as i64;
        }
    });

    pt_want.encode_vec_i64(&data, pt_k.into());

    res.decrypt(module, &mut pt, &sk_dft, scratch.borrow());

    pt.sub_inplace_ab(module, &pt_want);

    let noise_have: f64 = pt.std().log2();

    assert!(
        noise_have < -((k_ct - base2k) as f64),
        "noise: {noise_have}"
    );
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
