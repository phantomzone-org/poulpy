use std::collections::HashMap;

use poulpy_hal::{
    api::{
        ScratchAvailable, TakeMatZnx, TakeSlice, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, TakeVecZnxDftSlice, TakeVecZnxSlice,
        VecZnxAddInplace, VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy, VecZnxDftAddInplace,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNegateInplace,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ToOwnedDeep},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use poulpy_core::{
    GLWEOperations, TakeGGLWE, TakeGLWECt,
    layouts::{Dsize, GGLWECiphertextLayout, GGSWInfos, GLWEInfos, LWEInfos},
};

use poulpy_core::glwe_packing;
use poulpy_core::layouts::{GGSW, GLWECiphertext, LWECiphertext, prepared::AutomorphismKeyPrepared};

use crate::tfhe::{
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKeyPrepared, LookUpTable, LookUpTableRotationDirection,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute},
};

impl<D: DataRef, BRA: BlindRotationAlgo, B> CirtuitBootstrappingExecute<B> for CircuitBootstrappingKeyPrepared<D, BRA, B>
where
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubInplace
        + VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotate
        + VecZnxNormalize<B>,
    B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    Scratch<B>: TakeVecZnx
        + TakeVecZnxDftSlice<B>
        + TakeVecZnxBig<B>
        + TakeVecZnxDft<B>
        + TakeMatZnx
        + ScratchAvailable
        + TakeVecZnxSlice
        + TakeSlice,
    BlindRotationKeyPrepared<D, BRA, B>: BlincRotationExecute<B>,
{
    fn execute_to_constant<DM: DataMut, DR: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GGSW<DM>,
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
        res: &mut GGSW<DM>,
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

#[allow(clippy::too_many_arguments)]
pub fn circuit_bootstrap_core<DRes, DLwe, DBrk, BRA: BlindRotationAlgo, B>(
    to_exponent: bool,
    module: &Module<B>,
    log_gap_out: usize,
    res: &mut GGSW<DRes>,
    lwe: &LWECiphertext<DLwe>,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyPrepared<DBrk, BRA, B>,
    scratch: &mut Scratch<B>,
) where
    DRes: DataMut,
    DLwe: DataRef,
    DBrk: DataRef,
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubInplace
        + VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxBigAllocBytes
        + VecZnxDftAddInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxRotate
        + VecZnxNormalize<B>,
    B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    Scratch<B>: TakeVecZnxDftSlice<B>
        + TakeVecZnxBig<B>
        + TakeVecZnxDft<B>
        + TakeVecZnx
        + ScratchAvailable
        + TakeVecZnxSlice
        + TakeMatZnx
        + TakeSlice,
    BlindRotationKeyPrepared<DBrk, BRA, B>: BlincRotationExecute<B>,
{
    #[cfg(debug_assertions)]
    {
        use poulpy_core::layouts::LWEInfos;

        assert_eq!(res.n(), key.brk.n());
        assert_eq!(lwe.base2k(), key.brk.base2k());
        assert_eq!(res.base2k(), key.brk.base2k());
    }

    let n: usize = res.n().into();
    let base2k: usize = res.base2k().into();
    let dnum: usize = res.dnum().into();
    let rank: usize = res.rank().into();
    let k: usize = res.k().into();

    let alpha: usize = dnum.next_power_of_two();

    let mut f: Vec<i64> = vec![0i64; (1 << log_domain) * alpha];

    if to_exponent {
        (0..dnum).for_each(|i| {
            f[i] = 1 << (base2k * (dnum - 1 - i));
        });
    } else {
        (0..1 << log_domain).for_each(|j| {
            (0..dnum).for_each(|i| {
                f[j * alpha + i] = j as i64 * (1 << (base2k * (dnum - 1 - i)));
            });
        });
    }

    // Lut precision, basically must be able to hold the decomposition power basis of the GGSW
    let mut lut: LookUpTable = LookUpTable::alloc(module, base2k, base2k * dnum, extension_factor);
    lut.set(module, &f, base2k * dnum);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    // TODO: separate GGSW k from output of blind rotation k
    let (mut res_glwe, scratch_1) = scratch.take_glwe_ct(res);

    let gglwe_infos: GGLWECiphertextLayout = GGLWECiphertextLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        dnum: dnum.into(),
        dsize: Dsize(1),
        rank_in: rank.max(1).into(),
        rank_out: rank.into(),
    };

    let (mut tmp_gglwe, scratch_2) = scratch_1.take_gglwe(&gglwe_infos);

    key.brk.execute(module, &mut res_glwe, lwe, &lut, scratch_2);

    let gap: usize = 2 * lut.drift / lut.extension_factor();

    let log_gap_in: usize = (usize::BITS - (gap * alpha - 1).leading_zeros()) as _;

    (0..dnum).for_each(|i| {
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
                scratch_2,
            );
        } else {
            tmp_glwe.trace(module, 0, module.log_n(), &res_glwe, &key.atk, scratch_2);
        }

        if i < dnum {
            res_glwe.rotate_inplace(module, -(gap as i64), scratch_2);
        }
    });

    // Expands GGLWE to GGSW using GGLWE(s^2)
    res.from_gglwe(module, &tmp_gglwe, &key.tsk, scratch_2);
}

#[allow(clippy::too_many_arguments)]
fn post_process<DataRes, DataA, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    a: &GLWECiphertext<DataA>,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &HashMap<i64, AutomorphismKeyPrepared<Vec<u8>, B>>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataA: DataRef,
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxIdftApplyTmpA<B>
        + VecZnxSub
        + VecZnxAddInplace
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxSubInplace
        + VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxDftApply<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxRotate
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    let log_n: usize = module.log_n();

    let mut cts: HashMap<usize, &mut GLWECiphertext<Vec<u8>>> = HashMap::new();

    // First partial trace, vanishes all coefficients which are not multiples of gap_in
    // [1, 1, 1, 1, 0, 0, 0, ..., 0, 0, -1, -1, -1, -1] -> [1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0]
    res.trace(
        module,
        module.log_n() - log_gap_in + 1,
        log_n,
        a,
        auto_keys,
        scratch,
    );

    // TODO: optimize with packing and final partial trace
    // If gap_out < gap_in, then we need to repack, i.e. reduce the cap between coefficients.
    if log_gap_in != log_gap_out {
        let steps: usize = 1 << log_domain;

        // TODO: from Scratch
        let mut cts_vec: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

        for i in 0..steps {
            if i != 0 {
                res.rotate_inplace(module, -(1 << log_gap_in), scratch);
            }
            cts_vec.push(res.to_owned_deep());
        }

        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            cts.insert(i * (1 << log_gap_out), ct);
        }

        glwe_packing(module, &mut cts, log_gap_out, auto_keys, scratch);
        let packed: &mut GLWECiphertext<Vec<u8>> = cts.remove(&0).unwrap();
        res.trace(
            module,
            log_n - log_gap_out,
            log_n,
            packed,
            auto_keys,
            scratch,
        );
    }
}
