use backend::{
    hal::{
        api::{
            MatZnxAlloc, ModuleNew, ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddNormal,
            VecZnxAddScalarInplace, VecZnxAlloc, VecZnxAllocBytes, VecZnxEncodeCoeffsi64, VecZnxFillUniform, VecZnxRotateInplace,
            VecZnxSub, VecZnxSwithcDegree, ZnxView,
        },
        layouts::{Backend, Module, ScratchOwned},
        oep::{
            ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl,
            TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
        },
    },
    implementation::cpu_spqlios::FFT64,
};
use sampling::source::Source;

use crate::{
    BlindRotationKeyCGGIExecLayoutFamily, CCGIBlindRotationFamily, GLWECiphertext, GLWEDecryptFamily, GLWEPlaintext, GLWESecret,
    GLWESecretExec, GLWESecretFamily, Infos, LWECiphertext, LWESecret,
    blind_rotation::{
        cggi::{cggi_blind_rotate, cggi_blind_rotate_scratch_space, negate_and_mod_switch_2n},
        key::{BlindRotationKeyCGGI, BlindRotationKeyCGGIExec},
        lut::LookUpTable,
    },
    lwe::{LWEPlaintext, ciphertext::LWECiphertextToRef},
};

#[test]
fn standard() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    blind_rotatio_test(&module, 224, 1, 1);
}

#[test]
fn block_binary() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    blind_rotatio_test(&module, 224, 7, 1);
}

#[test]
fn block_binary_extended() {
    let module: Module<FFT64> = Module::<FFT64>::new(512);
    blind_rotatio_test(&module, 224, 7, 2);
}

pub(crate) trait CGGITestModuleFamily<B: Backend> = CCGIBlindRotationFamily<B>
    + GLWESecretFamily<B>
    + GLWEDecryptFamily<B>
    + BlindRotationKeyCGGIExecLayoutFamily<B>
    + VecZnxAlloc
    + ScalarZnxAlloc
    + VecZnxFillUniform
    + VecZnxAddNormal
    + VecZnxAllocBytes
    + VecZnxAddScalarInplace
    + VecZnxEncodeCoeffsi64
    + VecZnxRotateInplace
    + VecZnxSwithcDegree
    + MatZnxAlloc
    + VecZnxSub;
pub(crate) trait CGGITestScratchFamily<B: Backend> = VecZnxDftAllocBytesImpl<B>
    + VecZnxBigAllocBytesImpl<B>
    + ScratchOwnedAllocImpl<B>
    + ScratchOwnedBorrowImpl<B>
    + TakeVecZnxDftImpl<B>
    + TakeVecZnxBigImpl<B>
    + TakeVecZnxDftSliceImpl<B>
    + ScratchAvailableImpl<B>
    + TakeVecZnxImpl<B>
    + TakeVecZnxSliceImpl<B>;

fn blind_rotatio_test<B: Backend>(module: &Module<B>, n_lwe: usize, block_size: usize, extension_factor: usize)
where
    Module<B>: CGGITestModuleFamily<B>,
    B: CGGITestScratchFamily<B>,
{
    let basek: usize = 19;

    let k_lwe: usize = 24;
    let k_brk: usize = 3 * basek;
    let rows_brk: usize = 2; // Ensures first limb is noise-free.
    let k_lut: usize = 1 * basek;
    let k_res: usize = 2 * basek;
    let rank: usize = 1;

    let message_modulus: usize = 1 << 4;

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_glwe_dft: GLWESecretExec<Vec<u8>, B> = GLWESecretExec::from(module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(BlindRotationKeyCGGI::generate_from_sk_scratch_space(
        module, basek, k_brk, rank,
    ));

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

    let mut brk: BlindRotationKeyCGGI<Vec<u8>> = BlindRotationKeyCGGI::alloc(module, n_lwe, basek, k_brk, rows_brk, rank);

    brk.generate_from_sk(
        module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        3.2,
        scratch.borrow(),
    );

    let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe);

    let x: i64 = 2;
    let bits: usize = 8;

    module.encode_coeff_i64(basek, &mut pt_lwe.data, 0, bits, 0, x, bits);

    lwe.encrypt_sk(
        module,
        &pt_lwe,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        3.2,
    );

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = 2 * (i as i64) + 1);

    let mut lut: LookUpTable = LookUpTable::alloc(module, basek, k_lut, extension_factor);
    lut.set(module, &f, message_modulus);

    let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_res, rank);

    let brk_exec: BlindRotationKeyCGGIExec<Vec<u8>, B> = BlindRotationKeyCGGIExec::from(module, &brk, scratch_br.borrow());

    cggi_blind_rotate(module, &mut res, &lwe, &lut, &brk_exec, scratch_br.borrow());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_res);

    res.decrypt(module, &mut pt_have, &sk_glwe_dft, scratch.borrow());

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space

    negate_and_mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe.to_ref());

    let pt_want: i64 = (lwe_2n[0]
        + lwe_2n[1..]
            .iter()
            .zip(sk_lwe.data.at(0, 0))
            .map(|(x, y)| x * y)
            .sum::<i64>())
        & (2 * lut.domain_size() - 1) as i64;

    lut.rotate(module, pt_want);

    // First limb should be exactly equal (test are parameterized such that the noise does not reach
    // the first limb)
    assert_eq!(pt_have.data.at(0, 0), lut.data[0].at(0, 0));
}
