use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptSk, GLWENoise, GLWEPublicKeyGenerate, GLWESub, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPublicKey, GLWEPublicKeyPreparedFactory, GLWESecret,
        GLWESecretPreparedFactory, LWEInfos,
        compressed::GLWECompressed,
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared},
    },
};

pub fn test_glwe_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniform + GLWESub,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k_ct: usize = 54;
    let k_pt: usize = 30;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

        let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_pt.into(),
        };

        let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> =
            ScratchOwned::alloc(GLWE::encrypt_sk_tmp_bytes(module, &glwe_infos).max(module.glwe_noise_tmp_bytes(&glwe_infos)));

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

        ct.encrypt_sk(
            module,
            &pt_want,
            &sk_prepared,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let noise_have: f64 = ct
            .noise(module, &pt_want, &sk_prepared, scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = SIGMA.log2() - (ct.k().as_usize() as f64) + 0.5;

        assert!(noise_have <= noise_want + 0.2);
    }
}

pub fn test_glwe_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWECompressedEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniform + GLWESub,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k_ct: usize = 54;
    let k_pt: usize = 30;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

        let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_pt.into(),
        };

        let mut ct_compressed: GLWECompressed<Vec<u8>> = GLWECompressed::alloc_from_infos(&glwe_infos);

        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            GLWECompressed::encrypt_sk_tmp_bytes(module, &glwe_infos).max(module.glwe_noise_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

        let seed_xa: [u8; 32] = [1u8; 32];

        ct_compressed.encrypt_sk(
            module,
            &pt_want,
            &sk_prepared,
            seed_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
        ct.decompress(module, &ct_compressed);

        let noise_have: f64 = ct
            .noise(module, &pt_want, &sk_prepared, scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = SIGMA.log2() - (ct.k().as_usize() as f64) + 0.5;
        assert!(noise_have <= noise_want + 0.2);
    }
}

pub fn test_glwe_encrypt_zero_sk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniform + GLWESub,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k_ct: usize = 54;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

        let pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_noise_tmp_bytes(&glwe_infos)
                .max(GLWE::encrypt_sk_tmp_bytes(module, &glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);

        ct.encrypt_zero_sk(
            module,
            &sk_prepared,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let noise_have: f64 = ct
            .noise(module, &pt, &sk_prepared, scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = SIGMA.log2() - (ct.k().as_usize() as f64) + 0.5;
        assert!(noise_have <= noise_want + 0.2);
    }
}

pub fn test_glwe_encrypt_pk<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptPk<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWESub,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k_ct: usize = 54;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

        let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut source_xu: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_noise_tmp_bytes(&glwe_infos)
                .max(GLWE::encrypt_pk_tmp_bytes(module, &glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
        sk_prepared.prepare(module, &sk);

        let mut pk: GLWEPublicKey<Vec<u8>> = GLWEPublicKey::alloc_from_infos(&glwe_infos);
        pk.generate(module, &sk_prepared, &mut source_xa, &mut source_xe);

        module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

        let mut pk_prepared: GLWEPublicKeyPrepared<Vec<u8>, BE> = GLWEPublicKeyPrepared::alloc_from_infos(module, &glwe_infos);
        pk_prepared.prepare(module, &pk);

        ct.encrypt_pk(
            module,
            &pt_want,
            &pk_prepared,
            &mut source_xu,
            &mut source_xe,
            scratch.borrow(),
        );

        let noise_have: f64 = ct
            .noise(module, &pt_want, &sk_prepared, scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = ((((rank as f64) + 1.0) * n as f64 * 0.5 * SIGMA * SIGMA).sqrt()).log2() - (k_ct as f64);
        assert!(noise_have <= noise_want + 0.2);
    }
}
