use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScalarZnx, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, VecZnxBigAllocBytesImpl, VecZnxDftAllocBytesImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWCiphertextLayout, GLWESecret,
        compressed::{Decompress, GGSWCompressed},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_ggsw_encrypt_sk<B: Backend>(module: &Module<B>)
where
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
    Module<B>: SvpPrepare<B>,
{
    let base2k: usize = 12;
    let k: usize = 54;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSW::encrypt_sk_scratch_space(module, &ggsw_infos));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            ct.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

            ct.assert_noise(module, &sk_prepared, &pt_scalar, noise_f);
        }
    }
}

pub fn test_ggsw_compressed_encrypt_sk<B>(module: &Module<B>)
where
    Module<B>: VecZnxDftBytesOf
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
        + SvpPPolBytesOf
        + SvpPPolAlloc<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAddScalarInplace
        + VecZnxBigBytesOf
        + VecZnxBigAddInplace<B>
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<B>,
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
    let k: usize = 54;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos: GGSWCiphertextLayout = GGSWCiphertextLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ct_compressed: GGSWCompressed<Vec<u8>> = GGSWCompressed::alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GGSWCompressed::encrypt_sk_scratch_space(
                module,
                &ggsw_infos,
            ));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);
            let sk_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch.borrow());

            let seed_xa: [u8; 32] = [1u8; 32];

            ct_compressed.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                seed_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);
            ct.decompress(module, &ct_compressed);

            ct.assert_noise(module, &sk_prepared, &pt_scalar, noise_f);
        }
    }
}
