use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy},
    layouts::{DeviceBuf, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWENoise, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyEncryptSk, ScratchTakeCore,
    decryption::GLWEDecrypt,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dsize, GGLWE, GGLWEDecompress, GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyCompressed, GGLWEToGGSWKeyDecompress,
        GGLWEToGGSWKeyLayout, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory,
        prepared::GLWESecretPrepared,
    },
};

pub fn test_gglwe_to_ggsw_key_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretTensorFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GGLWENoise<BE>
        + VecZnxCopy,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 2_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let key_infos = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        })
        .unwrap();

        let mut key: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GGLWEToGGSWKeyEncryptSk::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(
            module, &key_infos,
        ));

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        GGLWEToGGSWKeyEncryptSk::gglwe_to_ggsw_key_encrypt_sk(
            module,
            &mut key,
            &sk,
            &key_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc_from_infos(&sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, scratch.borrow());

        let max_noise = DEFAULT_SIGMA_XE.log2() + 0.5 - (k as f64);

        let mut pt_want: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                module.vec_znx_copy(&mut pt_want.as_vec_znx_mut(), j, &sk_tensor.at(i, j).as_vec_znx(), 0);
            }

            let ksk: &GGLWE<Vec<u8>> = key.at(i);
            for row in 0..ksk.dnum().as_usize() {
                for col in 0..ksk.rank_in().as_usize() {
                    let noise_have = ksk
                        .noise(module, row, col, &pt_want, &sk_prepared, scratch.borrow())
                        .std()
                        .log2();
                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                }
            }
        }
    }
}

pub fn test_gglwe_to_ggsw_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGLWEToGGSWKeyCompressedEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWESecretTensorFactory<BE>
        + GGLWENoise<BE>
        + GGLWEDecompress
        + GGLWEToGGSWKeyDecompress,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let key_infos = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        })
        .unwrap();

        let mut key_compressed: GGLWEToGGSWKeyCompressed<Vec<u8>> = GGLWEToGGSWKeyCompressed::alloc_from_infos(&key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            GGLWEToGGSWKeyCompressedEncryptSk::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(module, &key_infos),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let seed_xa: [u8; 32] = [1u8; 32];

        GGLWEToGGSWKeyCompressedEncryptSk::gglwe_to_ggsw_key_encrypt_sk(
            module,
            &mut key_compressed,
            &sk,
            seed_xa,
            &key_infos,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut key: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&key_infos);
        module.decompress_gglwe_to_ggsw_key(&mut key, &key_compressed);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc_from_infos(&sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, scratch.borrow());

        let max_noise = DEFAULT_SIGMA_XE.log2() + 0.5 - (k as f64);

        let mut pt_want: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                module.vec_znx_copy(&mut pt_want.as_vec_znx_mut(), j, &sk_tensor.at(i, j).as_vec_znx(), 0);
            }

            let ksk: &GGLWE<Vec<u8>> = key.at(i);
            for row in 0..ksk.dnum().as_usize() {
                for col in 0..ksk.rank_in().as_usize() {
                    let noise_have = ksk
                        .noise(module, row, col, &pt_want, &sk_prepared, scratch.borrow())
                        .std()
                        .log2();
                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                }
            }
        }
    }
}
