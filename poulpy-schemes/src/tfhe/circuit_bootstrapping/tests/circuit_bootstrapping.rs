use std::time::Instant;

use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, ScalarZnx, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::tfhe::{
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, BlindRotationKeyLayout},
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory, CirtuitBootstrappingExecute,
    },
};

use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk, ScratchTakeCore,
    layouts::{
        Dsize, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout, GLWESecretPreparedFactory, GLWETensorKeyLayout,
        LWELayout,
    },
};

use poulpy_core::layouts::{
    GGSW, GLWE, GLWEPlaintext, GLWESecret, LWE, LWEPlaintext, LWESecret,
    prepared::{GGSWPrepared, GLWESecretPrepared},
};

pub fn test_circuit_bootstrapping_to_exponent<BE: Backend, M, BRA: BlindRotationAlgo>(module: &M)
where
    M: ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CirtuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateInplace<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, // TODO find a way to remove this bound or move it to CBT KEY
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 17;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 4;
    let k_lwe_ct: usize = 22;
    let block_size: usize = 7;

    let k_brk: usize = 5 * base2k;
    let rows_brk: usize = 4;

    let k_atk: usize = 5 * base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = 5 * base2k;
    let rows_tsk: usize = 4;

    let k_ggsw_res: usize = 4 * base2k;
    let rows_ggsw_res: usize = 2;

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_atk.into(),
            dnum: rows_atk.into(),
            rank: rank.into(),
            dsize: Dsize(1),
        },
        layout_tsk: GLWETensorKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: Dsize(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe.into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
    sk_glwe_prepared.prepare(module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(base2k.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(&cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let now: Instant = Instant::now();
    cbt_key.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

    let log_gap_out = 1;

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_exponent(
        module,
        log_gap_out,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    pt_ggsw.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(
        data * (1 << log_gap_out),
        &mut pt_ggsw.as_vec_znx_mut(),
        0,
        scratch.borrow(),
    );

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (base2k - 2);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &res);
    res_prepared.prepare(module, &res, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_prepared, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ggsw_infos);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data.at(0, 0)[0];
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}

pub fn test_circuit_bootstrapping_to_constant<BE: Backend, M, BRA: BlindRotationAlgo>(module: &M)
where
    M: ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CirtuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateInplace<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, // TODO find a way to remove this bound or move it to CBT KEY
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 14;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 1;
    let k_lwe_ct: usize = 13;
    let block_size: usize = 7;

    let k_brk: usize = 5 * base2k;
    let rows_brk: usize = 3;

    let k_atk: usize = 5 * base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = 5 * base2k;
    let rows_tsk: usize = 4;

    let k_ggsw_res: usize = 4 * base2k;
    let rows_ggsw_res: usize = 3;

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        layout_brk: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        layout_atk: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_atk.into(),
            dnum: rows_atk.into(),
            rank: rank.into(),
            dsize: Dsize(1),
        },
        layout_tsk: GLWETensorKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: Dsize(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe.into(), rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
    sk_glwe_prepared.prepare(module, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(base2k.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);
    ct_lwe.encrypt_sk(module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe);

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(&cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let now: Instant = Instant::now();
    cbt_key.encrypt_sk(
        module,
        &sk_lwe,
        &sk_glwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_constant(
        module,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n_glwe, 1);
    pt_ggsw.at_mut(0, 0)[0] = data;

    res.print_noise(module, &sk_glwe_prepared, &pt_ggsw);

    let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (base2k - k_lwe_pt - 1);

    ct_glwe.encrypt_sk(
        module,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(module, &res);
    res_prepared.prepare(module, &res, scratch.borrow());

    ct_glwe.external_product_inplace(module, &res_prepared, scratch.borrow());

    let mut pt_res: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ggsw_infos);
    ct_glwe.decrypt(module, &mut pt_res, &sk_glwe_prepared, scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
