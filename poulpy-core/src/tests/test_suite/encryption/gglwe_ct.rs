use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolAllocBytes, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
        VecZnxSubScalarInplace, VecZnxSwitchRing, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGLWECiphertextLayout, GGLWESwitchingKey, GLWESecret,
        compressed::{Decompress, GGLWESwitchingKeyCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_gglwe_switching_key_encrypt_sk<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxSwitchRing
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 54;
    let dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let n: usize = module.n();
                let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                let gglwe_infos: GGLWECiphertextLayout = GGLWECiphertextLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(&gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKey::encrypt_sk_scratch_space(
                    module,
                    &gglwe_infos,
                ));

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

                ksk.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                ksk.key
                    .assert_noise(module, &sk_out_prepared, &sk_in.data, SIGMA);
            }
        }
    }
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
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
        + VecZnxBigAddSmallInplace<B>
        + VecZnxSwitchRing
        + VecZnxAddScalarInplace
        + VecZnxBigAllocBytes
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxBigImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxImpl<B>
        + VecZnxDftAllocBytesImpl<B>
        + VecZnxBigAllocBytesImpl<B>
        + TakeSvpPPolImpl<B>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 54;
    let dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let n: usize = module.n();
                let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                let gglwe_infos: GGLWECiphertextLayout = GGLWECiphertextLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk_compressed: GGLWESwitchingKeyCompressed<Vec<u8>> = GGLWESwitchingKeyCompressed::alloc(&gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGLWESwitchingKeyCompressed::encrypt_sk_scratch_space(
                    module,
                    &gglwe_infos,
                ));

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_with(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

                let seed_xa = [1u8; 32];

                ksk_compressed.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    seed_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let mut ksk: GGLWESwitchingKey<Vec<u8>> = GGLWESwitchingKey::alloc(&gglwe_infos);
                ksk.decompress(module, &ksk_compressed);

                ksk.key
                    .assert_noise(module, &sk_out_prepared, &sk_in.data, SIGMA);
            }
        }
    }
}
