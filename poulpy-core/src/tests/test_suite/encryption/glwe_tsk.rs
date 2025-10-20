use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxCopy, VecZnxDftAlloc, VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyTmpA, VecZnxSubScalarInplace,
        VecZnxSwitchRing,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned, VecZnxBig, VecZnxDft},
    source::Source,
};

use crate::{
    GGLWETensorKeyCompressedEncryptSk, ScratchTakeCore, TensorKeyEncryptSk,
    decryption::GLWEDecrypt,
    encryption::SIGMA,
    layouts::{
        Dsize, GLWEPlaintext, GLWESecret, GLWESecretPrepare, GLWESecretPreparedAlloc, TensorKey, TensorKeyCompressed,
        TensorKeyLayout, prepared::GLWESecretPrepared,
    },
};

pub fn test_gglwe_tensor_key_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: TensorKeyEncryptSk<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + GLWEDecrypt<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxSubScalarInplace,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k: usize = 54;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos = TensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        };

        let mut tensor_key: TensorKey<Vec<u8>> = TensorKey::alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(TensorKey::encrypt_sk_tmp_bytes(module, &tensor_key_infos));

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&tensor_key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        tensor_key.encrypt_sk(
            module,
            &sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&tensor_key_infos);

        let mut sk_ij_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(1, 1);
        let mut sk_ij_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, 1);
        let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), 1_u32.into());
        let mut sk_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(rank, 1);

        for i in 0..rank {
            module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        }

        for i in 0..rank {
            for j in 0..rank {
                module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_prepared.data, j, &sk_dft, i);
                module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(
                    base2k,
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
                    base2k,
                    &sk_ij_big,
                    0,
                    scratch.borrow(),
                );
                for row_i in 0..dnum {
                    let ct = tensor_key.at(i, j).at(row_i, 0);

                    ct.decrypt(module, &mut pt, &sk_prepared, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, 0);

                    let std_pt: f64 = pt.data.std(base2k, 0) * (k as f64).exp2();
                    assert!((SIGMA - std_pt).abs() <= 0.5, "{SIGMA} {std_pt}");
                }
            }
        }
    }
}

pub fn test_gglwe_tensor_key_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: TensorKeyEncryptSk<BE>
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + GGLWETensorKeyCompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxSubScalarInplace
        + VecZnxFillUniform
        + VecZnxCopy
        + VecZnxSwitchRing,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k = 8;
    let k = 54;
    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos: TensorKeyLayout = TensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        };

        let mut tensor_key_compressed: TensorKeyCompressed<Vec<u8>> = TensorKeyCompressed::alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(TensorKeyCompressed::encrypt_sk_tmp_bytes(
            module,
            &tensor_key_infos,
        ));

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&tensor_key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        let seed_xa: [u8; 32] = [1u8; 32];

        tensor_key_compressed.encrypt_sk(module, &sk, seed_xa, &mut source_xe, scratch.borrow());

        let mut tensor_key: TensorKey<Vec<u8>> = TensorKey::alloc_from_infos(&tensor_key_infos);
        tensor_key.decompress(module, &tensor_key_compressed);

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&tensor_key_infos);

        let mut sk_ij_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(1, 1);
        let mut sk_ij_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, 1);
        let mut sk_ij: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), 1_u32.into());
        let mut sk_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(rank, 1);

        for i in 0..rank {
            module.vec_znx_dft_apply(1, 0, &mut sk_dft, i, &sk.data.as_vec_znx(), i);
        }

        for i in 0..rank {
            for j in 0..rank {
                module.svp_apply_dft_to_dft(&mut sk_ij_dft, 0, &sk_prepared.data, j, &sk_dft, i);
                module.vec_znx_idft_apply_tmpa(&mut sk_ij_big, 0, &mut sk_ij_dft, 0);
                module.vec_znx_big_normalize(
                    base2k,
                    &mut sk_ij.data.as_vec_znx_mut(),
                    0,
                    base2k,
                    &sk_ij_big,
                    0,
                    scratch.borrow(),
                );
                for row_i in 0..dnum {
                    tensor_key
                        .at(i, j)
                        .at(row_i, 0)
                        .decrypt(module, &mut pt, &sk_prepared, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(&mut pt.data, 0, row_i, &sk_ij.data, 0);

                    let std_pt: f64 = pt.data.std(base2k, 0) * (k as f64).exp2();
                    assert!((SIGMA - std_pt).abs() <= 0.5, "{SIGMA} {std_pt}");
                }
            }
        }
    }
}
