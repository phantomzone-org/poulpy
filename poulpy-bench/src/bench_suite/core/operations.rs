use poulpy_core::{
    GLWEAdd, GLWEMulPlain, GLWENormalize, GLWESub, ScratchTakeCore,
    layouts::{GLWE, GLWEInfos, GLWEPlaintext},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};
use std::{hint::black_box};

use criterion::Criterion;

pub fn bench_glwe_add<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEAdd,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_add::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_add(&mut res, &a, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_add_inplace<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEAdd,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_add_inplace::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_add_inplace(&mut res, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_sub<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWESub,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_sub::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_sub(&mut res, &a, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_sub_inplace<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWESub,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_sub_inplace::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_sub_inplace(&mut res, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_normalize<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    let group_name = format!("glwe_normalize::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_normalize(&mut res, &a, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_normalize_inplace<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    let group_name = format!("glwe_normalize_inplace::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_normalize_inplace(&mut res, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_mul_plain<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(infos, 0, infos, infos));

    let group_name = format!("glwe_mul_plain::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_mul_plain(&mut ct_out, 0, &ct_in, &pt, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_mul_plain_inplace<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(infos, 0, infos, infos));

    let group_name = format!("glwe_mul_plain_inplace::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_mul_plain_inplace(&mut ct, 0, &pt, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
