use super::CKKSTestParams;
use crate::{
    encoding::classical::{decode, encode},
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext},
    leveled::encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, SIGMA, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

pub struct SetupResult<BE: Backend> {
    pub sk: GLWESecretPrepared<Vec<u8>, BE>,
    pub re1: Vec<f64>,
    pub im1: Vec<f64>,
    pub re2: Vec<f64>,
    pub im2: Vec<f64>,
}

pub fn setup<BE: Backend>(module: &Module<BE>, params: CKKSTestParams) -> SetupResult<BE>
where
    Module<BE>: ModuleN + GLWESecretPreparedFactory<BE>,
{
    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);

    let glwe_infos = GLWELayout {
        n: Degree(n as u32),
        base2k,
        k,
        rank: Rank(1),
    };

    let mut source_xs = Source::new([0u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_infos);
    sk.fill_ternary_hw(params.hw, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(module, &glwe_infos);
    sk_prepared.prepare(module, &sk);

    let re1 = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
    let im1 = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();
    let re2 = (0..m).map(|i| 0.3 * (1.0 - (i as f64) / (m as f64))).collect();
    let im2 = (0..m).map(|i| -0.2 * (i as f64) / (m as f64)).collect();

    SetupResult {
        sk: sk_prepared,
        re1,
        im1,
        re2,
        im2,
    }
}

pub fn alloc_scratch<BE: Backend>(module: &Module<BE>, params: CKKSTestParams) -> ScratchOwned<BE>
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(module, &ct).max(decrypt_tmp_bytes(module, &ct)))
}

pub struct EncryptEnv<'a, BE: Backend> {
    pub module: &'a Module<BE>,
    pub params: CKKSTestParams,
    pub scratch_be: &'a mut ScratchOwned<BE>,
}

pub fn encrypt<BE: Backend>(
    env: &mut EncryptEnv<'_, BE>,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    re: &[f64],
    im: &[f64],
    source_xa_seed: [u8; 32],
    source_xe_seed: [u8; 32],
) -> CKKSCiphertext<Vec<u8>>
where
    Module<BE>: ModuleN + GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = env.module.n();
    let base2k = Base2K(env.params.base2k);
    let k = TorusPrecision(env.params.k);
    let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, env.params.log_delta);
    let mut ct = CKKSCiphertext::alloc(Degree(n as u32), base2k, k, env.params.log_delta);
    let mut source_xa = Source::new(source_xa_seed);
    let mut source_xe = Source::new(source_xe_seed);
    encode(&mut pt, re, im);
    encrypt_sk(
        env.module,
        &mut ct,
        &pt,
        sk,
        &mut source_xa,
        &mut source_xe,
        env.scratch_be.borrow(),
    );
    ct
}

pub fn decrypt_decode<BE: Backend>(
    module: &Module<BE>,
    params: CKKSTestParams,
    ct: &CKKSCiphertext<Vec<u8>>,
    sk: &GLWESecretPrepared<Vec<u8>, BE>,
    scratch_be: &mut ScratchOwned<BE>,
) -> (Vec<f64>, Vec<f64>)
where
    Module<BE>: ModuleN + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n = module.n();
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let mut pt_out = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, params.log_delta);
    decrypt(module, &mut pt_out, ct, sk, scratch_be.borrow());
    decode(&pt_out)
}

pub fn tol(n: usize, log_delta: u32) -> f64 {
    2.0 * (n as f64) * 6.0 * SIGMA / (1u64 << log_delta) as f64
}
