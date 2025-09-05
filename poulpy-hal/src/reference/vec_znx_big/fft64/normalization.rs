use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToRef, VecZnxToMut, ZnxView, ZnxViewMut},
    oep::VecZnxBigAllocBytesImpl,
    reference::vec_znx::vec_znx_normalize_ref,
    source::Source,
};

pub fn vec_znx_big_normalize_ref<R, A, BE>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxBigToRef<BE>,
    BE: Backend<ScalarBig = i64>,
{
    let a: VecZnxBig<&[u8], _> = a.to_ref();
    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_normalize_ref(basek, res, res_col, &a_vznx, a_col, carry);
}

pub fn test_vec_znx_big_normalize<B>(module: &Module<B>)
where
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    Module<B>: VecZnxBigNormalize<B> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek: usize = 12;

    let mut carry: Vec<i64> = vec![0i64; module.n()];

    let mut scratch = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            // Reference
            for i in 0..cols {
                vec_znx_big_normalize_ref(basek, &mut res_0, i, &a, i, &mut carry);
                module.vec_znx_big_normalize(basek, &mut res_1, i, &a, i, scratch.borrow());
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn bench_vec_znx_normalize<B>(c: &mut Criterion, label: &str)
where
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
        Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let basek = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        res.fill_uniform(&mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_big_normalize(basek, &mut res, i, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
