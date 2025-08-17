use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAllocBytes, VecZnxDftFromVecZnx,
        VecZnxDftToVecZnxBigConsume, VecZnxFillUniform, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes, VmpPMatAlloc, VmpPrepare,
        ZnxView,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
};
use sampling::source::Source;

use crate::layouts::{
    Infos, LWECiphertext, LWEPlaintext, LWESecret, LWESwitchingKey,
    prepared::{LWESwitchingKeyPrepared, PrepareAlloc},
};

pub fn test_lwe_keyswitch<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftFromVecZnx<B>
        + SvpApplyInplace<B>
        + VecZnxDftToVecZnxBigConsume<B>
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
        + VmpApplyTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxAutomorphismInplace,
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
    let n: usize = module.n();
    let basek: usize = 17;
    let sigma: f64 = 3.2;

    let n_lwe_in: usize = 22;
    let n_lwe_out: usize = 30;
    let k_lwe_ct: usize = 2 * basek;
    let k_lwe_pt: usize = 8;

    let k_ksk: usize = k_lwe_ct + basek;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k_ksk)
            | LWECiphertext::keyswitch_scratch_space(module, n, basek, k_lwe_ct, k_lwe_ct, k_ksk),
    );

    let mut sk_lwe_in: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_in);
    sk_lwe_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe_out: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_out);
    sk_lwe_out.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt_in: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_pt);

    lwe_pt_in.encode_i64(data, k_lwe_pt);

    let mut lwe_ct_in: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe_in, basek, k_lwe_ct);
    lwe_ct_in.encrypt_sk(
        module,
        &lwe_pt_in,
        &sk_lwe_in,
        &mut source_xa,
        &mut source_xe,
        sigma,
    );

    let mut ksk: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc(n, basek, k_ksk, lwe_ct_in.size());

    ksk.encrypt_sk(
        module,
        &sk_lwe_in,
        &sk_lwe_out,
        &mut source_xa,
        &mut source_xe,
        sigma,
        scratch.borrow(),
    );

    let mut lwe_ct_out: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe_out, basek, k_lwe_ct);

    let ksk_prepared: LWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    lwe_ct_out.keyswitch(module, &lwe_ct_in, &ksk_prepared, scratch.borrow());

    let mut lwe_pt_out: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(basek, k_lwe_ct);
    lwe_ct_out.decrypt(module, &mut lwe_pt_out, &sk_lwe_out);

    assert_eq!(lwe_pt_in.data.at(0, 0)[0], lwe_pt_out.data.at(0, 0)[0]);
}
