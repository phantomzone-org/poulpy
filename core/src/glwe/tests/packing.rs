use std::collections::HashMap;

use backend::hal::{
    api::{
        MatZnxAlloc, ScalarZnxAlloc, ScalarZnxAllocBytes, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarInplace,
        VecZnxAlloc, VecZnxAllocBytes, VecZnxAutomorphism, VecZnxBigSubSmallBInplace, VecZnxEncodeVeci64, VecZnxRotateInplace,
        VecZnxStd, VecZnxSwithcDegree,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::{
    AutomorphismKey, AutomorphismKeyExec, GGLWEExecLayoutFamily, GLWECiphertext, GLWEDecryptFamily, GLWEKeyswitchFamily, GLWEOps,
    GLWEPacker, GLWEPackingFamily, GLWEPlaintext, GLWESecret, GLWESecretExec, GLWESecretFamily, GLWESwitchingKeyEncryptSkFamily,
};

pub(crate) trait PackingTestModuleFamily<B: Backend> = GLWEPackingFamily<B>
    + GLWESecretFamily<B>
    + GLWESwitchingKeyEncryptSkFamily<B>
    + GLWEKeyswitchFamily<B>
    + GLWEDecryptFamily<B>
    + GGLWEExecLayoutFamily<B>
    + MatZnxAlloc
    + VecZnxAlloc
    + ScalarZnxAlloc
    + ScalarZnxAllocBytes
    + VecZnxAllocBytes
    + VecZnxStd
    + VecZnxSwithcDegree
    + VecZnxAddScalarInplace
    + VecZnxEncodeVeci64
    + VecZnxRotateInplace
    + VecZnxAutomorphism
    + VecZnxBigSubSmallBInplace<B>;

pub(crate) trait PackingTestScratchFamily<B: Backend> = TakeVecZnxDftImpl<B>
    + TakeVecZnxBigImpl<B>
    + TakeSvpPPolImpl<B>
    + ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>
    + ScratchAvailableImpl<B>
    + TakeScalarZnxImpl<B>
    + TakeVecZnxImpl<B>;

pub(crate) fn test_packing<B: Backend>(module: &Module<B>)
where
    Module<B>: PackingTestModuleFamily<B>,
    B: PackingTestScratchFamily<B>,
{
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let basek: usize = 18;
    let k_ct: usize = 36;
    let pt_k: usize = 18;
    let rank: usize = 3;
    let sigma: f64 = 3.2;
    let digits: usize = 1;
    let k_ksk: usize = k_ct + basek * digits;

    let rows: usize = k_ct.div_ceil(basek * digits);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct)
            | AutomorphismKey::encrypt_sk_scratch_space(module, basek, k_ksk, rank)
            | GLWEPacker::scratch_space(module, basek, k_ct, k_ksk, digits, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let sk_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);
    let mut data: Vec<i64> = vec![0i64; module.n()];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    module.encode_vec_i64(basek, &mut pt.data, 0, pt_k, &data, 32);

    let gal_els: Vec<i64> = GLWEPacker::galois_elements(module);

    let mut auto_keys: HashMap<i64, AutomorphismKeyExec<Vec<u8>, B>> = HashMap::new();
    let mut tmp: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc(module, basek, k_ksk, rows, digits, rank);
    gal_els.iter().for_each(|gal_el| {
        tmp.encrypt_sk(
            module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );
        let atk_exec: AutomorphismKeyExec<Vec<u8>, B> = AutomorphismKeyExec::from(module, &tmp, scratch.borrow());
        auto_keys.insert(*gal_el, atk_exec);
    });

    let log_batch: usize = 0;

    let mut packer: GLWEPacker = GLWEPacker::new(module, log_batch, basek, k_ct, rank);

    let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);

    ct.encrypt_sk(
        module,
        &pt,
        &sk_dft,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let log_n: usize = module.log_n();

    (0..module.n() >> log_batch).for_each(|i| {
        ct.encrypt_sk(
            module,
            &pt,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        pt.rotate_inplace(module, -(1 << log_batch)); // X^-batch * pt

        if reverse_bits_msb(i, log_n as u32) % 5 == 0 {
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

    let mut res = GLWECiphertext::alloc(module, basek, k_ct, rank);
    packer.flush(module, &mut res);

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_ct);
    let mut data: Vec<i64> = vec![0i64; module.n()];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        if i % 5 == 0 {
            *x = reverse_bits_msb(i, log_n as u32) as i64;
        }
    });

    module.encode_vec_i64(basek, &mut pt_want.data, 0, pt_k, &data, 32);

    res.decrypt(module, &mut pt, &sk_dft, scratch.borrow());

    pt.sub_inplace_ab(module, &pt_want);

    let noise_have: f64 = module.vec_znx_std(basek, &pt.data, 0).log2();
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
