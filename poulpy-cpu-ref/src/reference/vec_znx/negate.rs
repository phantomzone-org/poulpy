use crate::{
    api::{ModuleNew, VecZnxNegateBackend, VecZnxNegateInplaceBackend},
    layouts::{
        Backend, FillUniform, Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, VecZnxToMut, VecZnxToRef, ZnxInfos,
        ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxNegate, ZnxNegateInplace, ZnxZero},
    source::Source,
};

pub fn vec_znx_negate<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxNegate + ZnxZero,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        ZNXARI::znx_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_assign<R, ZNXARI>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxNegateAssign,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        ZNXARI::znx_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn bench_vec_znx_negate<B: Backend<OwnedBuf = Vec<u8>>>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegateBackend<B> + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegateBackend<B> + ModuleNew<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_negate_backend(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    i,
                );
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

pub fn bench_vec_znx_negate_inplace<B: Backend<OwnedBuf = Vec<u8>>>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegateInplaceBackend<B> + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegateInplaceBackend<B> + ModuleNew<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        move || {
            for i in 0..cols {
                module.vec_znx_negate_inplace_backend(&mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut a), i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
