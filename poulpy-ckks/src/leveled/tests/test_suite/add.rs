use super::{
    CKKSTestParams,
    helpers::{EncryptEnv, SetupResult, alloc_scratch, decrypt_decode, encrypt, setup, tol},
};
use crate::{
    encoding::classical::encode,
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::operations::add::{add, add_const, add_const_inplace, add_inplace, add_pt, add_pt_inplace},
};
use poulpy_core::{
    GLWEAdd, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWESecretPreparedFactory, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

pub fn test_add<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
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
    let ct2 = encrypt(
        &mut EncryptEnv {
            module,
            params,
            scratch_be: &mut scratch_be,
        },
        &sk,
        &re2,
        &im2,
        [3u8; 32],
        [4u8; 32],
    );

    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    add(module, &mut ct_res, &ct1, &ct2);
    let (re_out, im_out) = decrypt_decode(module, params, &ct_res, &sk, &mut scratch_be);

    let t = 2.0 * tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] + re2[j])).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] + re2[j]
        );
        assert!(
            (im_out[j] - (im1[j] + im2[j])).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] + im2[j]
        );
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
    add_inplace(module, &mut ct_ip, &ct2);
    let (re_ip, im_ip) = decrypt_decode(module, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_pt<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, re2, im2 } = setup(module, params);
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
    let mut pt2 = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    encode(&mut pt2, &re2, &im2);

    let mut ct_res = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    add_pt(module, &mut ct_res, &ct1, &pt2);
    let (re_out, im_out) = decrypt_decode(module, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!((re_out[j] - (re1[j] + re2[j])).abs() < t, "re[{j}]");
        assert!((im_out[j] - (im1[j] + im2[j])).abs() < t, "im[{j}]");
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
    add_pt_inplace(module, &mut ct_ip, &pt2);
    let (re_ip, im_ip) = decrypt_decode(module, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + re2[j])).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + im2[j])).abs() < t, "inplace im[{j}]");
    }
}

pub fn test_add_const<BE: Backend>(module: &Module<BE>, params: CKKSTestParams)
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let m = n / 2;
    let SetupResult { sk, re1, im1, .. } = setup(module, params);
    let mut scratch_be = alloc_scratch(module, params);
    let (c_re, c_im) = (0.1, -0.05);

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
    add_const(module, &mut ct_res, &ct1, c_re, c_im);
    let (re_out, im_out) = decrypt_decode(module, params, &ct_res, &sk, &mut scratch_be);

    let t = tol(n, params.log_delta);
    for j in 0..m {
        assert!(
            (re_out[j] - (re1[j] + c_re)).abs() < t,
            "re[{j}]: {} vs {}",
            re_out[j],
            re1[j] + c_re
        );
        assert!(
            (im_out[j] - (im1[j] + c_im)).abs() < t,
            "im[{j}]: {} vs {}",
            im_out[j],
            im1[j] + c_im
        );
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
    add_const_inplace(module, &mut ct_ip, c_re, c_im);
    let (re_ip, im_ip) = decrypt_decode(module, params, &ct_ip, &sk, &mut scratch_be);
    for j in 0..m {
        assert!((re_ip[j] - (re1[j] + c_re)).abs() < t, "inplace re[{j}]");
        assert!((im_ip[j] - (im1[j] + c_im)).abs() < t, "inplace im[{j}]");
    }
}
