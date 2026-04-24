use poulpy_core::{
    GLWEAdd, GLWEMulPlain, GLWENormalize, GLWESub,
    layouts::{GLWE, GLWEInfos, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
};
use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_add_into<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEAdd,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_add_into::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_add_into(&mut res, &a, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_add_assign<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEAdd,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_add_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_add_assign(&mut res, &b);
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

pub fn bench_glwe_sub_assign<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWESub,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let group_name = format!("glwe_sub_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_sub_assign(&mut res, &b);
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_normalize<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
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
            let mut res_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut res);
            let a_backend = <GLWE<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(&a);
            module.glwe_normalize(&mut res_backend, &a_backend, &mut scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_normalize_inplace<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    let group_name = format!("glwe_normalize_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            let mut res_backend = <GLWE<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut res);
            module.glwe_normalize_inplace(&mut res_backend, &mut scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_mul_plain<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(infos, infos, infos));

    let group_name = format!("glwe_mul_plain::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_mul_plain(
                0,
                &mut ct_out,
                &ct_in,
                ct_in.max_k().as_usize(),
                &pt,
                pt.max_k().as_usize(),
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_mul_plain_inplace<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(infos, infos, infos));

    let group_name = format!("glwe_mul_plain_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_mul_plain_assign(
                0,
                &mut ct,
                infos.max_k().as_usize(),
                &pt,
                pt.max_k().as_usize(),
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}
