use std::collections::HashMap;

use poulpy_hal::{
    api::{
        DFT, IDFTConsume, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAllocBytes, VecZnxFillUniform,
        VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
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
        GGLWEAutomorphismKey, GLWECiphertext, GLWEPlaintext, GLWESecret,
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
        + VecZnxRshInplace
        + VecZnxRotateInplace
        + VecZnxBigNormalize<B>
        + DFT<B>
        + VecZnxRotate
        + SvpApplyInplace<B>
        + IDFTConsume<B>
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
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace
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
    let basek: usize = 18;
    let k_ct: usize = 36;
    let pt_k: usize = 18;
    let rank: usize = 3;
    let digits: usize = 1;
    let k_ksk: usize = k_ct + basek * digits;

    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct)
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWEPacker::scratch_space(module, basek, k_ct, k_ksk, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    pt.encode_vec_i64(&data, pt_k);

    let gal_els: Vec<i64> = GLWEPacker::galois_elements(module);

    let mut auto_keys: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>> = HashMap::new();
    let mut tmp: GGLWEAutomorphismKey<Vec<u8>> = GGLWEAutomorphismKey::alloc(n, basek, k_ksk, rows, digits, rank);
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

    let mut packer: GLWEPacker = GLWEPacker::new(n, log_batch, basek, k_ct, rank);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct, rank);

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

        pt.rotate_inplace(module, -(1 << log_batch)); // X^-batch * pt

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

    let mut res = GLWECiphertext::alloc(n, basek, k_ct, rank);
    packer.flush(module, &mut res);

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(n, basek, k_ct);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        if i.is_multiple_of(5) {
            *x = reverse_bits_msb(i, log_n as u32) as i64;
        }
    });

    pt_want.encode_vec_i64(&data, pt_k);

    res.decrypt(module, &mut pt, &sk_dft, scratch.borrow());

    pt.sub_inplace_ab(module, &pt_want);

    let noise_have: f64 = pt.std().log2();
    // println!("noise_have: {}", noise_have);
    assert!(
        noise_have < -((k_ct - basek) as f64),
        "noise: {}",
        noise_have
    );
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
