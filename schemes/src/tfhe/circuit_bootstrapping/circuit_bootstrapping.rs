use std::{collections::HashMap, usize};

use backend::hal::{
    api::{
        ScratchAvailable, TakeMatZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, TakeVecZnxDftSlice, TakeVecZnxSlice,
        VecZnxAddInplace, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy, VecZnxDftAddInplace,
        VecZnxDftAllocBytes, VecZnxDftCopy, VecZnxDftFromVecZnx, VecZnxDftToVecZnxBigConsume, VecZnxDftToVecZnxBigTmpA,
        VecZnxNegateInplace, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubABInplace, VecZnxSwithcDegree, VmpApply, VmpApplyAdd, VmpApplyTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use core::{GLWEOperations, TakeGGLWE, TakeGLWECt, layouts::Infos};

use core::layouts::{GGSWCiphertext, GLWECiphertext, LWECiphertext, prepared::GGLWEAutomorphismKeyPrepared};

use crate::tfhe::{
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKeyPrepared, LookUpTable, LookUpTableRotationDirection,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute},
};

impl<D: DataRef, BRA: BlindRotationAlgo, B: Backend> CirtuitBootstrappingExecute<B> for CircuitBootstrappingKeyPrepared<D, BRA, B>
where
    Module<B>: VecZnxRotateInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace
        + VecZnxDftCopy<B>
        + VecZnxDftToVecZnxBigTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftFromVecZnx<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotate,
    B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    Scratch<B>: TakeVecZnx
        + TakeVecZnxDftSlice<B>
        + TakeVecZnxBig<B>
        + TakeVecZnxDft<B>
        + TakeMatZnx
        + ScratchAvailable
        + TakeVecZnxSlice,
    BlindRotationKeyPrepared<D, BRA, B>: BlincRotationExecute<B>,
{
    fn execute_to_constant<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GGSWCiphertext<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    ) {
        circuit_bootstrap_core(
            false,
            module,
            0,
            res,
            lwe,
            log_domain,
            extension_factor,
            self,
            scratch,
        );
    }

    fn execute_to_exponent<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        log_gap_out: usize,
        res: &mut GGSWCiphertext<DM>,
        lwe: &LWECiphertext<DR>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<B>,
    ) {
        circuit_bootstrap_core(
            true,
            module,
            log_gap_out,
            res,
            lwe,
            log_domain,
            extension_factor,
            self,
            scratch,
        );
    }
}

pub fn circuit_bootstrap_core<DRes, DLwe, DBrk, BRA: BlindRotationAlgo, B: Backend>(
    to_exponent: bool,
    module: &Module<B>,
    log_gap_out: usize,
    res: &mut GGSWCiphertext<DRes>,
    lwe: &LWECiphertext<DLwe>,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyPrepared<DBrk, BRA, B>,
    scratch: &mut Scratch<B>,
) where
    DRes: DataMut,
    DLwe: DataRef,
    DBrk: DataRef,
    Module<B>: VecZnxRotateInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace
        + VecZnxDftCopy<B>
        + VecZnxDftToVecZnxBigTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftFromVecZnx<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotate,
    B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    Scratch<B>: TakeVecZnxDftSlice<B>
        + TakeVecZnxBig<B>
        + TakeVecZnxDft<B>
        + TakeVecZnx
        + ScratchAvailable
        + TakeVecZnxSlice
        + TakeMatZnx,
    BlindRotationKeyPrepared<DBrk, BRA, B>: BlincRotationExecute<B>,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), key.brk.n());
        assert_eq!(lwe.basek(), key.brk.basek());
        assert_eq!(res.basek(), key.brk.basek());
    }

    let n: usize = res.n();
    let basek: usize = res.basek();
    let rows: usize = res.rows();
    let rank: usize = res.rank();
    let k: usize = res.k();

    let alpha: usize = rows.next_power_of_two();

    let mut f: Vec<i64> = vec![0i64; (1 << log_domain) * alpha];

    if to_exponent {
        (0..rows).for_each(|i| {
            f[i] = 1 << (basek * (rows - 1 - i));
        });
    } else {
        (0..1 << log_domain).for_each(|j| {
            (0..rows).for_each(|i| {
                f[j * alpha + i] = j as i64 * (1 << (basek * (rows - 1 - i)));
            });
        });
    }

    // Lut precision, basically must be able to hold the decomposition power basis of the GGSW
    let mut lut: LookUpTable = LookUpTable::alloc(n, basek, basek * rows, extension_factor);
    lut.set(module, &f, basek * rows);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    // TODO: separate GGSW k from output of blind rotation k
    let (mut res_glwe, scratch1) = scratch.take_glwe_ct(n, basek, k, rank);
    let (mut tmp_gglwe, scratch2) = scratch1.take_gglwe(n, basek, k, rows, 1, rank.max(1), rank);

    key.brk.execute(module, &mut res_glwe, &lwe, &lut, scratch2);

    let gap: usize = 2 * lut.drift / lut.extension_factor();

    let log_gap_in: usize = (usize::BITS - (gap * alpha - 1).leading_zeros()) as _;

    (0..rows).for_each(|i| {
        let mut tmp_glwe: GLWECiphertext<&mut [u8]> = tmp_gglwe.at_mut(i, 0);

        if to_exponent {
            // Isolates i-th LUT and moves coefficients according to requested gap.
            post_process(
                module,
                &mut tmp_glwe,
                &res_glwe,
                log_gap_in,
                log_gap_out,
                log_domain,
                &key.atk,
                scratch2,
            );
        } else {
            tmp_glwe.trace(module, 0, module.log_n(), &res_glwe, &key.atk, scratch2);
        }

        if i < rows {
            res_glwe.rotate_inplace(module, -(gap as i64));
        }
    });

    // Expands GGLWE to GGSW using GGLWE(s^2)
    res.from_gglwe(module, &tmp_gglwe, &key.tsk, scratch2);
}

fn post_process<DataRes, DataA, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    a: &GLWECiphertext<DataA>,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataA: DataRef,
    Module<B>: VecZnxRotateInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace
        + VecZnxDftCopy<B>
        + VecZnxDftToVecZnxBigTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftFromVecZnx<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRotate,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    let log_n: usize = module.log_n();

    let mut cts: HashMap<usize, GLWECiphertext<Vec<u8>>> = HashMap::new();

    // First partial trace, vanishes all coefficients which are not multiples of gap_in
    // [1, 1, 1, 1, 0, 0, 0, ..., 0, 0, -1, -1, -1, -1] -> [1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0]
    res.trace(
        module,
        module.log_n() - log_gap_in as usize + 1,
        log_n,
        &a,
        auto_keys,
        scratch,
    );

    // TODO: optimize with packing and final partial trace
    // If gap_out < gap_in, then we need to repack, i.e. reduce the cap between coefficients.
    if log_gap_in != log_gap_out {
        let steps: i32 = 1 << log_domain;
        (0..steps).for_each(|i| {
            if i != 0 {
                res.rotate_inplace(module, -(1 << log_gap_in));
            }
            cts.insert(i as usize * (1 << log_gap_out), res.clone());
        });
        pack(module, &mut cts, log_gap_out, auto_keys, scratch);
        let packed: GLWECiphertext<Vec<u8>> = cts.remove(&0).unwrap();
        res.trace(
            module,
            log_n - log_gap_out,
            log_n,
            &packed,
            auto_keys,
            scratch,
        );
    }
}

pub fn pack<D: DataMut, B: Backend>(
    module: &Module<B>,
    cts: &mut HashMap<usize, GLWECiphertext<D>>,
    log_gap_out: usize,
    auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxRotateInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace
        + VecZnxDftCopy<B>
        + VecZnxDftToVecZnxBigTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftFromVecZnx<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRotate,
    Scratch<B>: TakeVecZnx + TakeVecZnxDft<B> + ScratchAvailable,
{
    let log_n: usize = module.log_n();

    let basek: usize = cts.get(&0).unwrap().basek();
    let k: usize = cts.get(&0).unwrap().k();
    let rank: usize = cts.get(&0).unwrap().rank();

    (0..log_n - log_gap_out).for_each(|i| {
        let t = 16.min(1 << (log_n - 1 - i));

        let auto_key: &GGLWEAutomorphismKeyPrepared<Vec<u8>, B>;
        if i == 0 {
            auto_key = auto_keys.get(&-1).unwrap()
        } else {
            auto_key = auto_keys.get(&module.galois_element(1 << (i - 1))).unwrap();
        }

        (0..t).for_each(|j| {
            let mut a: Option<GLWECiphertext<D>> = cts.remove(&j);
            let mut b: Option<GLWECiphertext<D>> = cts.remove(&(j + t));

            combine(
                module,
                basek,
                k,
                rank,
                a.as_mut(),
                b.as_mut(),
                i,
                auto_key,
                scratch,
            );

            if let Some(a) = a {
                cts.insert(j, a);
            } else if let Some(b) = b {
                cts.insert(j, b);
            }
        });
    });
}

fn combine<A: DataMut, D: DataMut, DataAK: DataRef, B: Backend>(
    module: &Module<B>,
    basek: usize,
    k: usize,
    rank: usize,
    a: Option<&mut GLWECiphertext<A>>,
    b: Option<&mut GLWECiphertext<D>>,
    i: usize,
    auto_key: &GGLWEAutomorphismKeyPrepared<DataAK, B>,
    scratch: &mut Scratch<B>,
) where
    Module<B>: VecZnxRotateInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwithcDegree
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace
        + VecZnxDftCopy<B>
        + VecZnxDftToVecZnxBigTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubABInplace
        + VecZnxDftAllocBytes
        + VmpApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApply<B>
        + VmpApplyAdd<B>
        + VecZnxDftFromVecZnx<B>
        + VecZnxDftToVecZnxBigConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRotate,
    Scratch<B>: TakeVecZnx + TakeVecZnxDft<B> + ScratchAvailable,
{
    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    // Different cases for wether a and/or b are zero.
    //
    // Implicite RSH without modulus switch, introduces extra I(X) * Q/2 on decryption.
    // Necessary so that the scaling of the plaintext remains constant.
    // It however is ok to do so here because coefficients are eventually
    // either mapped to garbage or twice their value which vanishes I(X)
    // since 2*(I(X) * Q/2) = I(X) * Q = 0 mod Q.
    if let Some(a) = a {
        let n: usize = a.n();
        let log_n: usize = (u64::BITS - (n - 1).leading_zeros()) as _;
        let t: i64 = 1 << (log_n - i - 1);

        if let Some(b) = b {
            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(n, basek, k, rank);

            // a = a * X^-t
            a.rotate_inplace(module, -t);

            // tmp_b = a * X^-t - b
            tmp_b.sub(module, a, b);
            tmp_b.rsh(module, 1);

            // a = a * X^-t + b
            a.add_inplace(module, b);
            a.rsh(module, 1);

            tmp_b.normalize_inplace(module, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            tmp_b.automorphism_inplace(module, auto_key, scratch_1);

            // a = a * X^-t + b - phi(a * X^-t - b)
            a.sub_inplace_ab(module, &tmp_b);
            a.normalize_inplace(module, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            a.rotate_inplace(module, t);
        } else {
            a.rsh(module, 1);
            // a = a + phi(a)
            a.automorphism_add_inplace(module, auto_key, scratch);
        }
    } else {
        if let Some(b) = b {
            let n: usize = b.n();
            let log_n: usize = (u64::BITS - (n - 1).leading_zeros()) as _;
            let t: i64 = 1 << (log_n - i - 1);

            let (mut tmp_b, scratch_1) = scratch.take_glwe_ct(n, basek, k, rank);
            tmp_b.rotate(module, t, b);
            tmp_b.rsh(module, 1);

            // a = (b* X^t - phi(b* X^t))
            b.automorphism_sub_ba(module, &tmp_b, auto_key, scratch_1);
        }
    }
}
