use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    decryption::GLWEDecrypt, encryption::SIGMA, layouts::{
        prepared::{GGLWEPrepare, GGLWEPreparedAlloc, GLWESecretPrepared}, GGLWELayout, GLWESecret, GLWESecretPrepare, GLWESecretPreparedAlloc, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyCompressed, GLWESwitchingKeyDecompress
    }, noise::GGLWENoise, GGLWEEncryptSk, GGLWEKeyswitch, GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptSk, ScratchTakeCore
};

pub fn test_gglwe_switching_key_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGLWEEncryptSk<BE>
        + GGLWEPrepare<BE>
        + GGLWEPreparedAlloc<BE>
        + GGLWEKeyswitch<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedAlloc<BE>
        + GLWESecretPrepare<BE>
        + GLWESwitchingKeyAlloc
        + GLWESwitchingKeyEncryptSk<BE>
        + VecZnxFillUniform
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 54;
    let dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let n: usize = module.n();
                let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                let gglwe_infos: GGLWELayout = GGLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(module, &gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWESwitchingKey::encrypt_sk_tmp_bytes(
                    module,
                    &gglwe_infos,
                ));

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk_out_prepared.prepare(module, &sk_out);

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

pub fn test_gglwe_switching_key_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGLWEEncryptSk<BE>
        + GGLWEPrepare<BE>
        + GGLWEPreparedAlloc<BE>
        + GGLWEKeyswitch<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedAlloc<BE>
        + GLWESecretPrepare<BE>
        + GLWESwitchingKeyAlloc
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + GLWESwitchingKeyDecompress
        + GGLWENoise<BE>
        + VecZnxFillUniform,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_ksk: usize = 54;
    let dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let n: usize = module.n();
                let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                let gglwe_infos: GGLWELayout = GGLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk_compressed: GLWESwitchingKeyCompressed<Vec<u8>> =
                    GLWESwitchingKeyCompressed::alloc_from_infos(module, &gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWESwitchingKeyCompressed::encrypt_sk_tmp_bytes(
                    module,
                    &gglwe_infos,
                ));

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(module, rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk_out_prepared.prepare(module, &sk_out);

                let seed_xa = [1u8; 32];

                ksk_compressed.encrypt_sk(
                    module,
                    &sk_in,
                    &sk_out,
                    seed_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(module, &gglwe_infos);
                ksk.decompress(module, &ksk_compressed);

                ksk.key
                    .assert_noise(module, &sk_out_prepared, &sk_in.data, SIGMA);
            }
        }
    }
}
