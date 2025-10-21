use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxFillUniform},
    layouts::{Backend, GaloisElement, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    AutomorphismKeyCompressedEncryptSk, AutomorphismKeyEncryptSk, GGLWEKeyswitch, GLWESwitchingKeyCompressedEncryptSk,
    GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        AutomorphismKey, AutomorphismKeyDecompress, AutomorphismKeyLayout, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        GLWESwitchingKeyDecompress, compressed::GLWEAutomorphismKeyCompressed, prepared::GLWESecretPrepared,
    },
    noise::GGLWENoise,
};

pub fn test_gglwe_automorphism_key_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: AutomorphismKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + GLWESwitchingKeyDecompress
        + GGLWENoise<BE>
        + VecZnxFillUniform
        + VecZnxAutomorphism,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
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

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(AutomorphismKey::encrypt_sk_tmp_bytes(module, &atk_infos));

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
            let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, sk_out.rank().into());
            sk_out_prepared.prepare(module, &sk_out);

            atk.key
                .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
        }
    }
}

pub fn test_gglwe_automorphism_key_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: AutomorphismKeyCompressedEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + AutomorphismKeyDecompress
        + VecZnxAutomorphism
        + VecZnxFillUniform
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
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

            let mut atk_compressed: GLWEAutomorphismKeyCompressed<Vec<u8>> =
                GLWEAutomorphismKeyCompressed::alloc_from_infos(&atk_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWEAutomorphismKeyCompressed::encrypt_sk_tmp_bytes(
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
            let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, sk_out.rank().into());
            sk_out_prepared.prepare(module, &sk_out);

            let mut atk: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&atk_infos);
            atk.decompress(module, &atk_compressed);

            atk.key
                .assert_noise(module, &sk_out_prepared, &sk.data, SIGMA);
        }
    }
}
