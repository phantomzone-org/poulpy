use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GGLWENoise, GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk, ScratchTakeCore,
    decryption::GLWEDecrypt,
    encryption::SIGMA,
    layouts::{
        Dsize, GGLWEDecompress, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor,
        GLWESecretTensorFactory, GLWETensorKey, GLWETensorKeyCompressed, GLWETensorKeyLayout, LWEInfos,
        prepared::GLWESecretPrepared,
    },
};

pub fn test_gglwe_tensor_key_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWETensorKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWESecretTensorFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k: usize = 54;

    for rank in 2_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos = GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        };

        let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWETensorKey::encrypt_sk_tmp_bytes(
            module,
            &tensor_key_infos,
        ));

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

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc_from_infos(&sk);
        sk_tensor.prepare(module, &sk, scratch.borrow());

        let max_noise: f64 = SIGMA.log2() - (tensor_key.k().as_usize() as f64) + 0.5;

        for row in 0..tensor_key.dnum().as_usize() {
            for col in 0..tensor_key.rank().as_usize() + 1 {
                assert!(
                    tensor_key
                        .0
                        .noise(
                            module,
                            row,
                            col,
                            &sk_tensor.data,
                            &sk_prepared,
                            scratch.borrow()
                        )
                        .std()
                        .log2()
                        <= max_noise
                )
            }
        }
    }
}

pub fn test_gglwe_tensor_key_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWETensorKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensorKeyCompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretTensorFactory<BE>
        + GGLWENoise<BE>
        + GGLWEDecompress,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k = 8;
    let k = 54;
    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos: GLWETensorKeyLayout = GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        };

        let mut tensor_key_compressed: GLWETensorKeyCompressed<Vec<u8>> =
            GLWETensorKeyCompressed::alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWETensorKeyCompressed::encrypt_sk_tmp_bytes(
            module,
            &tensor_key_infos,
        ));

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&tensor_key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        let seed_xa: [u8; 32] = [1u8; 32];

        tensor_key_compressed.encrypt_sk(module, &sk, seed_xa, &mut source_xe, scratch.borrow());

        let mut tensor_key: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc_from_infos(&tensor_key_infos);
        tensor_key.decompress(module, &tensor_key_compressed);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc_from_infos(&sk);
        sk_tensor.prepare(module, &sk, scratch.borrow());

        let max_noise: f64 = SIGMA.log2() - (tensor_key.k().as_usize() as f64) + 0.5;

        for row in 0..tensor_key.dnum().as_usize() {
            for col in 0..tensor_key.rank().as_usize() + 1 {
                assert!(
                    tensor_key
                        .0
                        .noise(
                            module,
                            row,
                            col,
                            &sk_tensor.data,
                            &sk_prepared,
                            scratch.borrow()
                        )
                        .std()
                        .log2()
                        <= max_noise
                )
            }
        }
    }
}
