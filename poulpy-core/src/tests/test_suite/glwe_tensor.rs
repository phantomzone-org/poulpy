use std::f64::consts::SQRT_2;

use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxSubInplace,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned, VecZnx},
    source::Source,
    test_suite::convolution::bivariate_convolution_naive,
};

use crate::{
    GLWEDecrypt, GLWEEncryptSk, GLWETensoring, ScratchTakeCore,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretTensorPrepared, GLWETensor, LWEInfos, TorusPrecision, prepared::GLWESecretPrepared,
    },
};

pub fn test_glwe_tensoring<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWETensoring<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + VecZnxSubInplace
        + VecZnxNormalizeInplace<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxCopy
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let in_base2k: usize = 16;
    let out_base2k: usize = 13;
    let k: usize = 128;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res: GLWETensor<Vec<u8>> = GLWETensor::alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            GLWE::encrypt_sk_tmp_bytes(module, &glwe_in_infos)
                .max(GLWE::decrypt_tmp_bytes(module, &glwe_out_infos))
                .max(module.glwe_tensor_tmp_bytes(&res, 0, &a, &b, in_base2k))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into())),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
        sk_dft.prepare(module, &sk);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc(module.n().into(), rank.into());
        sk_tensor.prepare(module, &sk, scratch.borrow());

        let mut sk_tensor_prep: GLWESecretTensorPrepared<Vec<u8>, BE> = GLWESecretTensorPrepared::alloc(module, rank.into());
        sk_tensor_prep.prepare(module, &sk_tensor);

        let scale: usize = 2 * in_base2k;

        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 7) - 4;
        }

        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let mut pt_want_base2k_in = VecZnx::alloc(n, 1, pt_in.size());
        bivariate_convolution_naive(
            module,
            in_base2k,
            2,
            &mut pt_want_base2k_in,
            0,
            pt_in.data(),
            0,
            pt_in.data(),
            0,
            scratch.borrow(),
        );

        a.encrypt_sk(module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());
        b.encrypt_sk(module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());

        for res_offset in 0..scale {
            module.glwe_tensor(&mut res, scale + res_offset, &a, &b, scratch.borrow());

            res.decrypt(module, &mut pt_have, &sk_dft, &sk_tensor_prep, scratch.borrow());
            module.vec_znx_normalize(
                pt_want.data_mut(),
                out_base2k,
                res_offset as i64,
                0,
                &pt_want_base2k_in,
                in_base2k,
                0,
                scratch.borrow(),
            );

            module.vec_znx_sub_inplace(&mut pt_want.data, 0, &pt_have.data, 0);
            module.vec_znx_normalize_inplace(pt_want.base2k().as_usize(), &mut pt_want.data, 0, scratch.borrow());

            let noise_have: f64 = pt_want.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}
