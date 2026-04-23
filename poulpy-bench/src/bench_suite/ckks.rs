use std::hint::black_box;

use criterion::Criterion;
use poulpy_ckks::{
    CKKSMeta,
    layouts::CKKSCiphertext,
    leveled::operations::{
        composite::{CKKSDotProductOps, CKKSMulAddOps},
        mul::CKKSMulOps,
    },
    oep::CKKSImpl,
};
use poulpy_core::{
    ScratchTakeCore,
    layouts::{Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, Rank, TorusPrecision},
    oep::CoreImpl,
};
use poulpy_hal::{
    api::{ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DeviceBuf, Module, Scratch, ScratchOwned, ZnxViewMut},
    oep::HalImpl,
};

const N: usize = 1 << 15;
const BASE2K: usize = 52;
const K: usize = 728;
const LOG_DECIMAL: usize = 40;
const DSIZE: usize = 1;
const DOT_PRODUCT_TERMS: usize = 8;

pub trait CkksBenchBackend: Backend + CKKSImpl<Self> + CoreImpl<Self> + HalImpl<Self>
where
    Self: Sized,
{
}

impl<BE> CkksBenchBackend for BE where BE: Backend + CKKSImpl<BE> + CoreImpl<BE> + HalImpl<BE> {}

struct CkksBenchSetup<BE: CkksBenchBackend> {
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    ct_a: CKKSCiphertext<Vec<u8>>,
    ct_b: CKKSCiphertext<Vec<u8>>,
    ct_dst: CKKSCiphertext<Vec<u8>>,
    tsk: poulpy_core::layouts::GLWETensorKeyPrepared<DeviceBuf<BE>, BE>,
}

fn ckks_layout() -> GLWELayout {
    GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(K as u32),
        rank: Rank(1),
    }
}

fn ckks_meta() -> CKKSMeta {
    CKKSMeta {
        log_decimal: LOG_DECIMAL,
        log_hom_rem: K - LOG_DECIMAL,
    }
}

fn tsk_layout() -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision((K + DSIZE * BASE2K) as u32),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(K.div_ceil(DSIZE * BASE2K) as u32),
    }
}

fn setup<BE: CkksBenchBackend>() -> CkksBenchSetup<BE>
where
    Module<BE>: ModuleNew<BE> + GLWETensorKeyPreparedFactory<BE> + CKKSMulOps<BE> + CKKSMulAddOps<BE> + CKKSDotProductOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let module = Module::<BE>::new(N as u64);
    let ct_layout = ckks_layout();
    let tsk_layout = tsk_layout();
    let meta = ckks_meta();

    let mut ct_a = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    let mut ct_b = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    let mut ct_dst = CKKSCiphertext::alloc_from_infos(&ct_layout).unwrap();
    ct_a.meta = meta;
    ct_b.meta = meta;
    ct_dst.meta = meta;

    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let scratch_bytes = module
        .ckks_mul_tmp_bytes(&ct_layout, &tsk_layout)
        .max(module.ckks_mul_add_ct_tmp_bytes(&ct_layout, &tsk_layout))
        .max(module.ckks_dot_product_ct_tmp_bytes(DOT_PRODUCT_TERMS, &ct_layout, &tsk_layout));
    let scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    CkksBenchSetup {
        module,
        scratch,
        ct_a,
        ct_b,
        ct_dst,
        tsk,
    }
}

fn reset_dst(dst: &mut CKKSCiphertext<Vec<u8>>) {
    dst.data_mut().raw_mut().fill(0);
    dst.meta = ckks_meta();
}

pub fn bench_ckks_mul_ct<BE: CkksBenchBackend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensorKeyPreparedFactory<BE> + CKKSMulOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let CkksBenchSetup {
        module,
        mut scratch,
        ct_a,
        ct_b,
        mut ct_dst,
        tsk,
    } = setup::<BE>();

    let group_name = format!("ckks_operations::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function("mul_ct", |bench| {
        bench.iter(|| {
            reset_dst(&mut ct_dst);
            module
                .ckks_mul(&mut ct_dst, black_box(&ct_a), black_box(&ct_b), &tsk, scratch.borrow())
                .unwrap();
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_ckks_mul_add_ct<BE: CkksBenchBackend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensorKeyPreparedFactory<BE> + CKKSMulAddOps<BE> + CKKSMulOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let CkksBenchSetup {
        module,
        mut scratch,
        ct_a,
        ct_b,
        mut ct_dst,
        tsk,
    } = setup::<BE>();

    let group_name = format!("ckks_composite::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function("mul_add_ct", |bench| {
        bench.iter(|| {
            reset_dst(&mut ct_dst);
            module
                .ckks_mul_add_ct(&mut ct_dst, black_box(&ct_a), black_box(&ct_b), &tsk, scratch.borrow())
                .unwrap();
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_ckks_dot_product_ct<BE: CkksBenchBackend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWETensorKeyPreparedFactory<BE> + CKKSDotProductOps<BE> + CKKSMulOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let CkksBenchSetup {
        module,
        mut scratch,
        ct_a,
        ct_b,
        mut ct_dst,
        tsk,
    } = setup::<BE>();
    let a: Vec<&CKKSCiphertext<Vec<u8>>> = (0..DOT_PRODUCT_TERMS).map(|_| &ct_a).collect();
    let b: Vec<&CKKSCiphertext<Vec<u8>>> = (0..DOT_PRODUCT_TERMS).map(|_| &ct_b).collect();

    let group_name = format!("ckks_composite::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("dot_product_ct_{DOT_PRODUCT_TERMS}"), |bench| {
        bench.iter(|| {
            reset_dst(&mut ct_dst);
            module
                .ckks_dot_product_ct(
                    &mut ct_dst,
                    black_box(a.as_slice()),
                    black_box(b.as_slice()),
                    &tsk,
                    scratch.borrow(),
                )
                .unwrap();
            black_box(());
        })
    });
    group.finish();
}
