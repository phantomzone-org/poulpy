use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism, VecZnxAutomorphismInplace,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace, VecZnxSubScalarInplace, VecZnxSwitchRing,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxImpl,
    },
    source::Source,
};

use crate::{
    encryption::SIGMA,
    layouts::{
        AutomorphismKey, AutomorphismKeyLayout, GLWEInfos, GLWESecret,
        compressed::{AutomorphismKeyCompressed, Decompress},
        prepared::{GLWESecretPrepared, PrepareAlloc},
    },
};

pub fn test_gglwe_automorphisk_key_encrypt_sk<B>(module: &Module<B>)
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
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigBytesOf
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 60;
    let dsize: usize = k_ksk.div_ceil(base2k) - 1;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

            let atk_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut atk: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&atk_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(AutomorphismKey::encrypt_sk_scratch_space(
                module, &atk_infos,
            ));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&atk_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let p = -5;

            atk.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
            (0..atk.rank().into()).for_each(|i| {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            });
            let sk_out_prepared: GLWESecretPrepared<Vec<u8>, B> = sk_out.prepare_alloc(module, scratch.borrow());

            atk.key
                .key
                .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
        }
    }
}

pub fn test_gglwe_automorphisk_key_compressed_encrypt_sk<B>(module: &Module<B>)
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
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxAddScalarInplace
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigBytesOf
        + VecZnxBigAddInplace<B>
        + VecZnxSubScalarInplace
        + VecZnxCopy
        + VmpPMatAlloc<B>
        + VmpPrepare<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + ScratchAvailableImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeVecZnxDftImpl<B>
        + TakeVecZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 60;
    let dsize: usize = k_ksk.div_ceil(base2k) - 1;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

            let atk_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut atk_compressed: AutomorphismKeyCompressed<Vec<u8>> = AutomorphismKeyCompressed::alloc_from_infos(&atk_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(AutomorphismKey::encrypt_sk_scratch_space(
                module, &atk_infos,
            ));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&atk_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let p = -5;

            let seed_xa: [u8; 32] = [1u8; 32];

            atk_compressed.encrypt_sk(module, p, &sk, seed_xa, &mut source_xe, scratch.borrow());

            let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
            (0..atk_compressed.rank().into()).for_each(|i| {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p),
                    &mut sk_out.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            });
            let sk_out_prepared = sk_out.prepare_alloc(module, scratch.borrow());

            let mut atk: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&atk_infos);
            atk.decompress(module, &atk_compressed);

            atk.key
                .key
                .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
        }
    }
}
