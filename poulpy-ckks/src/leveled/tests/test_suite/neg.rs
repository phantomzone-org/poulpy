use super::{
    CKKSTestParams,
    helpers::{EncryptEnv, SetupResult, alloc_scratch, decrypt_decode, encrypt, setup, tol},
};
use crate::{
    layouts::ciphertext::CKKSCiphertext,
    leveled::operations::neg::{neg, neg_inplace},
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

pub fn test_neg<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd
        + VecZnxNegate
        + VecZnxNegateInplace,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, .. } = setup(module, params);
    let mut scratch_be = alloc_scratch(module, params);

    let ct1 = encrypt(
        &mut EncryptEnv {
            module,
            params,
            scratch_be: &mut scratch_be,
        },
        &sk,
        &re1,
        &im1,
        [1u8; 32],
        [2u8; 32],
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    neg(module, &mut ct_res, &ct1);
    let (re_out, im_out) = decrypt_decode(module, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!((re_out[j] + re1[j]).abs() < t, "re[{j}]: {} vs {}", re_out[j], -re1[j]);
        assert!((im_out[j] + im1[j]).abs() < t, "im[{j}]: {} vs {}", im_out[j], -im1[j]);
    }

    let mut ct_ip = encrypt(
        &mut EncryptEnv {
            module,
            params,
            scratch_be: &mut scratch_be,
        },
        &sk,
        &re1,
        &im1,
        [1u8; 32],
        [2u8; 32],
    );
    neg_inplace(module, &mut ct_ip);
    let (re_ip, im_ip) = decrypt_decode(module, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] + re1[j]).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] + im1[j]).abs() < t, "inplace im[{j}]");
    }
}
