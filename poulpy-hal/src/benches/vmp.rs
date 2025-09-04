use std::hint::black_box;

use crate::{
    api::{DFT, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDftToDft, VmpPMatAlloc, VmpPrepare},
    layouts::{Backend, FillUniform, MatZnx, Module, ScratchOwned, VecZnx, ZnxInfos},
    source::Source,
};
use criterion::{BenchmarkId, Criterion};

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpPrepare<B> + DFT<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpPrepare<B> + DFT<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(module.n(), rows, cols_in, cols_out, size);
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        for i in 0..mat.rows() {
            for j in 0..mat.cols_in() {
                mat.at_mut(i, j).fill_uniform(&mut source);
            }
        }

        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols_in, rows);

        // Fill a with random i64
        b.fill_uniform(&mut source);

        let mut b_dft: crate::layouts::VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_in, rows);

        for i in 0..cols_in {
            module.dft(1, 0, &mut b_dft, i, &b, i);
        }

        let mut res: crate::layouts::VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);

        let mut pmat: crate::layouts::VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        module.vmp_prepare(&mut pmat, &mat, scratch.borrow());

        move || {
            module.vmp_apply_dft_to_dft(&mut res, &b_dft, &pmat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[1],
            params[2],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
