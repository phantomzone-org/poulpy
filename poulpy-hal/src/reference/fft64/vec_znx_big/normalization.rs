use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes},
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToRef, VecZnxToMut},
    reference::{
        vec_znx::vec_znx_normalize,
        znx::{
            ZnxNormalizeFinalStep, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep,
            ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
        },
    },
    source::Source,
};

pub fn vec_znx_big_normalize_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_normalize<R, A, BE>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxBigToRef<BE>,
    BE: Backend<ScalarBig = i64>
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxZero,
{
    let a: VecZnxBig<&[u8], _> = a.to_ref();
    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_normalize::<_, _, BE>(basek, res, res_col, &a_vznx, a_col, carry);
}

pub fn bench_vec_znx_normalize<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let basek: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(res.data_mut());

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
