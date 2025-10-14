use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
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
    LWE, LWECiphertextLayout, LWEPlaintext, LWESecret, LWESwitchingKey, LWESwitchingKeyLayout,
    prepared::{LWESwitchingKeyPrepared, PrepareAlloc},
};

pub fn test_lwe_keyswitch<B>(module: &Module<B>)
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
        + ZnAddNormal
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
    let n: usize = module.n();
    let base2k: usize = 17;

    let n_lwe_in: usize = 22;
    let n_lwe_out: usize = 30;
    let k_lwe_ct: usize = 2 * base2k;
    let k_lwe_pt: usize = 8;

    let k_ksk: usize = k_lwe_ct + base2k;
    let dnum: usize = k_lwe_ct.div_ceil(base2k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let key_apply_infos: LWESwitchingKeyLayout = LWESwitchingKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_ksk.into(),
        dnum: dnum.into(),
    };

    let lwe_in_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe_in.into(),
        base2k: base2k.into(),
        k: k_lwe_ct.into(),
    };

    let lwe_out_infos: LWECiphertextLayout = LWECiphertextLayout {
        n: n_lwe_out.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        LWESwitchingKey::encrypt_sk_scratch_space(module, &key_apply_infos)
            | LWE::keyswitch_scratch_space(module, &lwe_out_infos, &lwe_in_infos, &key_apply_infos),
    );

    let mut sk_lwe_in: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_in.into());
    sk_lwe_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe_out: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_out.into());
    sk_lwe_out.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt_in: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(base2k.into(), k_lwe_pt.into());
    lwe_pt_in.encode_i64(data, k_lwe_pt.into());

    let mut lwe_ct_in: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_in_infos);
    lwe_ct_in.encrypt_sk(
        module,
        &lwe_pt_in,
        &sk_lwe_in,
        &mut source_xa,
        &mut source_xe,
    );

    let mut ksk: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc_from_infos(&key_apply_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe_in,
        &sk_lwe_out,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct_out: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_out_infos);

    let ksk_prepared: LWESwitchingKeyPrepared<Vec<u8>, B> = ksk.prepare_alloc(module, scratch.borrow());

    lwe_ct_out.keyswitch(module, &lwe_ct_in, &ksk_prepared, scratch.borrow());

    let mut lwe_pt_out: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_out_infos);
    lwe_ct_out.decrypt(module, &mut lwe_pt_out, &sk_lwe_out);

    assert_eq!(lwe_pt_in.data.at(0, 0)[0], lwe_pt_out.data.at(0, 0)[0]);
}
