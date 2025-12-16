use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxNormalizeInplace, VecZnxSubInplace},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWEDecrypt, GLWEEncryptSk, GLWETensoring, ScratchTakeCore,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretTensorPrepared, GLWETensor, TorusPrecision, prepared::GLWESecretPrepared,
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
        + GLWESecretTensorFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let in_base2k: usize = 10;
    let out_base2k: usize = 10;
    let k: usize = 54;

    for rank in 1_usize..2 {
        println!("rank: {rank}");

        let n: usize = module.n();

        let glwe_in_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: (k - in_base2k).into(),
            rank: rank.into(),
        };

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res: GLWETensor<Vec<u8>> = GLWETensor::alloc_from_infos(&glwe_out_infos);
        let mut pt_have_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_have_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        //let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        //let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let res_offset: usize = 0;

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            GLWE::encrypt_sk_tmp_bytes(module, &glwe_in_infos)
                .max(GLWE::decrypt_tmp_bytes(module, &glwe_out_infos))
                .max(module.glwe_tensor_tmp_bytes(&res, res_offset, &a, &b))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into())),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        // sk.fill_ternary_prob(0.5, &mut source_xs);
        sk.fill_zero();

        let mut sk_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
        sk_dft.prepare(module, &sk);

        let k: usize = 1;

        let mut data_want: Vec<i64> = vec![0i64; n];
        data_want[2 * k] = 1; // X^k * X^k = X^2k

        // X^k
        pt_have_in.encode_coeff_i64(255, TorusPrecision(3 * in_base2k as u32), k);

        a.encrypt_sk(module, &pt_have_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());

        b.encrypt_sk(module, &pt_have_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());

        module.glwe_tensor(&mut res, res_offset, &a, &b, scratch.borrow());

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc(module.n().into(), rank.into());
        sk_tensor.prepare(module, &sk, scratch.borrow());

        let mut sk_tensor_prep: GLWESecretTensorPrepared<Vec<u8>, BE> = GLWESecretTensorPrepared::alloc(module, rank.into());
        sk_tensor_prep.prepare(module, &sk_tensor);

        res.decrypt(module, &mut pt_have_out, &sk_dft, &sk_tensor_prep, scratch.borrow());

        println!("pt_have_out: {}", pt_have_out);

        // glwe_out.decrypt(module, &mut pt_have, &sk_dft, scratch.borrow());

        // module.vec_znx_sub_inplace(&mut pt_want.data, 0, &pt_have.data, 0);
        // module.vec_znx_normalize_inplace(
        //    pt_want.base2k().as_usize(),
        //    &mut pt_want.data,
        //    0,
        //    scratch.borrow(),
        //);
    }
}
