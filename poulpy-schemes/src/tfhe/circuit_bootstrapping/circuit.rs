use std::collections::HashMap;

use poulpy_hal::{
    api::{ModuleLogN, ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned, ToOwnedDeep},
};

use poulpy_core::{
    GGSWFromGGLWE, GLWEDecrypt, GLWEPacking, GLWERotate, GLWETrace, ScratchTakeCore,
    layouts::{
        Dsize, GGLWE, GGLWEInfos, GGLWELayout, GGLWEPreparedToRef, GGSWInfos, GGSWToMut, GLWEAutomorphismKeyHelper, GLWEInfos,
        GLWESecretPreparedFactory, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos, LWEToRef, Rank,
    },
};

use poulpy_core::layouts::{GGSW, GLWE, LWE};

use crate::tfhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, LookUpTableLayout, LookUpTableRotationDirection, LookupTable, LookupTableFactory,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared},
};

pub trait CirtuitBootstrappingExecute<BRA: BlindRotationAlgo, BE: Backend> {
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos;

    fn circuit_bootstrapping_execute_to_constant<R, L, D>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_execute_to_exponent<R, L, D>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef;
}

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingKeyPrepared<D, BRA, BE> {
    pub fn execute_to_constant<M, L, R>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        M: CirtuitBootstrappingExecute<BRA, BE>,
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_constant(res, lwe, self, log_domain, extension_factor, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_to_exponent<R, L, M>(
        &self,
        module: &M,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        M: CirtuitBootstrappingExecute<BRA, BE>,
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_exponent(
            log_gap_out,
            res,
            lwe,
            self,
            log_domain,
            extension_factor,
            scratch,
        );
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> CirtuitBootstrappingExecute<BRA, BE> for Module<BE>
where
    Self: ModuleN
        + LookupTableFactory
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWFromGGLWE<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWERotate<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        let gglwe_infos: GGLWELayout = GGLWELayout {
            n: res_infos.n(),
            base2k: res_infos.base2k(),
            k: res_infos.k(),
            dnum: res_infos.dnum(),
            dsize: Dsize(1),
            rank_in: res_infos.rank().max(Rank(1)),
            rank_out: res_infos.rank(),
        };

        self.blind_rotation_execute_tmp_bytes(
            block_size,
            extension_factor,
            res_infos,
            &cbt_infos.brk_infos(),
        )
        .max(self.glwe_trace_tmp_bytes(res_infos, res_infos, &cbt_infos.atk_infos()))
        .max(self.ggsw_from_gglwe_tmp_bytes(res_infos, &cbt_infos.tsk_infos()))
            + GLWE::bytes_of_from_infos(res_infos)
            + GGLWE::bytes_of_from_infos(&gglwe_infos)
    }

    fn circuit_bootstrapping_execute_to_constant<R, L, D>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef,
    {
        assert!(
            scratch.available() >= self.circuit_bootstrapping_execute_tmp_bytes(key.block_size(), extension_factor, res, key)
        );

        circuit_bootstrap_core(
            false,
            self,
            0,
            res,
            lwe,
            log_domain,
            extension_factor,
            key,
            scratch,
        );
    }

    fn circuit_bootstrapping_execute_to_exponent<R, L, D>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef,
    {
        assert!(
            scratch.available() >= self.circuit_bootstrapping_execute_tmp_bytes(key.block_size(), extension_factor, res, key)
        );

        circuit_bootstrap_core(
            true,
            self,
            log_gap_out,
            res,
            lwe,
            log_domain,
            extension_factor,
            key,
            scratch,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn circuit_bootstrap_core<R, L, D, M, BRA: BlindRotationAlgo, BE: Backend>(
    to_exponent: bool,
    module: &M,
    log_gap_out: usize,
    res: &mut R,
    lwe: &L,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
    scratch: &mut Scratch<BE>,
) where
    R: GGSWToMut,
    L: LWEToRef,
    D: DataRef,
    M: ModuleN
        + LookupTableFactory
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWFromGGLWE<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWERotate<BE>
        + ModuleLogN,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
    let lwe: &LWE<&[u8]> = &lwe.to_ref();

    assert_eq!(res.n(), key.brk.n());
    assert_eq!(lwe.base2k(), key.brk.base2k());
    assert_eq!(res.base2k(), key.brk.base2k());

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

    let lut_infos: LookUpTableLayout = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: (base2k * dnum).into(),
        base2k: base2k.into(),
    };

    // Lut precision, basically must be able to hold the decomposition power basis of the GGSW
    let mut lut: LookupTable = LookupTable::alloc(&lut_infos);
    lut.set(module, &f, base2k * dnum);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    // TODO: separate GGSW k from output of blind rotation k
    let (mut res_glwe, scratch_1) = scratch.take_glwe(res);

    let gglwe_infos: GGLWELayout = GGLWELayout {
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
        let mut tmp_glwe: GLWE<&mut [u8]> = tmp_gglwe.at_mut(i, 0);

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
            tmp_glwe.trace(module, 0, &res_glwe, &key.atk, scratch_2);
        }

        // let sk_glwe: &poulpy_core::layouts::GLWESecret<&[u8]> = &sk_glwe.to_ref();
        // let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, sk_glwe.rank());
        // let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&res_glwe);
        // res_glwe.decrypt(module, &mut pt, &sk_glwe_prepared, scratch_2);
        // println!("pt[{i}]: {}", pt);

        if i < dnum {
            module.glwe_rotate_inplace(-(gap as i64), &mut res_glwe, scratch_2);
        }
    });

    // Expands GGLWE to GGSW using GGLWE(s^2)
    res.from_gglwe(module, &tmp_gglwe, &key.tsk, scratch_2);
}

#[allow(clippy::too_many_arguments)]
fn post_process<R, A, M, H, K, BE: Backend>(
    module: &M,
    res: &mut R,
    a: &A,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &H,
    scratch: &mut Scratch<BE>,
) where
    R: GLWEToMut,
    A: GLWEToRef,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    M: ModuleLogN + GLWETrace<BE> + GLWEPacking<BE> + GLWERotate<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
    let a: &GLWE<&[u8]> = &a.to_ref();

    let mut cts: HashMap<usize, &mut GLWE<Vec<u8>>> = HashMap::new();

    // First partial trace, vanishes all coefficients which are not multiples of gap_in
    // [1, 1, 1, 1, 0, 0, 0, ..., 0, 0, -1, -1, -1, -1] -> [1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0]
    res.trace(
        module,
        module.log_n() - log_gap_in + 1,
        a,
        auto_keys,
        scratch,
    );

    // TODO: optimize with packing and final partial trace
    // If gap_out < gap_in, then we need to repack, i.e. reduce the cap between coefficients.
    if log_gap_in != log_gap_out {
        let steps: usize = 1 << log_domain;

        // TODO: from Scratch
        let mut cts_vec: Vec<GLWE<Vec<u8>>> = Vec::new();

        for i in 0..steps {
            if i != 0 {
                module.glwe_rotate_inplace(-(1 << log_gap_in), res, scratch);
            }
            cts_vec.push(res.to_owned_deep());
        }

        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            cts.insert(i * (1 << log_gap_out), ct);
        }

        module.glwe_pack(res, cts, log_gap_out, auto_keys, scratch);
    }
}
