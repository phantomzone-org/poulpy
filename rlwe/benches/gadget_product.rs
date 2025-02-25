use base2k::{
    FFT64, Infos, Module, Sampling, SvpPPolOps, VecZnx, VecZnxDft, VecZnxDftOps, VecZnxOps,
    VmpPMat, alloc_aligned_u8,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rlwe::{
    ciphertext::{Ciphertext, new_gadget_ciphertext},
    elem::ElemCommon,
    encryptor::{encrypt_grlwe_sk, encrypt_grlwe_sk_tmp_bytes},
    gadget_product::{gadget_product_core, gadget_product_tmp_bytes},
    keys::SecretKey,
    parameters::{Parameters, ParametersLiteral},
};
use sampling::source::Source;

fn bench_gadget_product_inplace(c: &mut Criterion) {
    fn runner<'a>(
        module: &'a Module,
        res_dft_0: &'a mut VecZnxDft,
        res_dft_1: &'a mut VecZnxDft,
        a: &'a VecZnx,
        a_cols: usize,
        b: &'a Ciphertext<VmpPMat>,
        b_cols: usize,
        tmp_bytes: &'a mut [u8],
    ) -> Box<dyn FnMut() + 'a> {
        Box::new(move || {
            gadget_product_core(
                module, res_dft_0, res_dft_1, a, a_cols, b, b_cols, tmp_bytes,
            );
        })
    }

    let mut b: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("gadget_product_inplace");

    for log_n in 10..11 {
        let params_lit: ParametersLiteral = ParametersLiteral {
            log_n: log_n,
            log_q: 32,
            log_p: 0,
            log_base2k: 16,
            log_scale: 20,
            xe: 3.2,
            xs: 128,
        };

        let params: Parameters = Parameters::new::<FFT64>(&params_lit);

        let mut tmp_bytes: Vec<u8> = alloc_aligned_u8(
            params.encrypt_rlwe_sk_tmp_bytes(params.log_q())
                | gadget_product_tmp_bytes(
                    params.module(),
                    params.log_base2k(),
                    params.log_q(),
                    params.log_q(),
                    params.cols_q(),
                    params.log_qp(),
                )
                | encrypt_grlwe_sk_tmp_bytes(
                    params.module(),
                    params.log_base2k(),
                    params.cols_qp(),
                    params.log_qp(),
                ),
        );

        let mut source: Source = Source::new([3; 32]);

        let mut sk0: SecretKey = SecretKey::new(params.module());
        let mut sk1: SecretKey = SecretKey::new(params.module());
        sk0.fill_ternary_hw(params.xs(), &mut source);
        sk1.fill_ternary_hw(params.xs(), &mut source);

        let mut source_xe: Source = Source::new([4; 32]);
        let mut source_xa: Source = Source::new([5; 32]);

        let mut sk0_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
        params.module().svp_prepare(&mut sk0_svp_ppol, &sk0.0);

        let mut sk1_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
        params.module().svp_prepare(&mut sk1_svp_ppol, &sk1.0);

        let mut gadget_ct: Ciphertext<VmpPMat> = new_gadget_ciphertext(
            params.module(),
            params.log_base2k(),
            params.cols_q(),
            params.log_qp(),
        );

        encrypt_grlwe_sk(
            params.module(),
            &mut gadget_ct,
            &sk0.0,
            &sk1_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        let mut ct: Ciphertext<VecZnx> = params.new_ciphertext(params.log_q());

        params.encrypt_rlwe_sk_thread_safe(
            &mut ct,
            None,
            &sk0_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            &mut tmp_bytes,
        );

        let mut res_dft_0: VecZnxDft = params.module().new_vec_znx_dft(gadget_ct.cols());
        let mut res_dft_1: VecZnxDft = params.module().new_vec_znx_dft(gadget_ct.cols());

        let mut a: VecZnx = params.module().new_vec_znx(params.cols_q());
        params
            .module()
            .fill_uniform(params.log_base2k(), &mut a, params.cols_q(), &mut source_xa);

        let a_cols: usize = a.cols();
        let b_cols: usize = gadget_ct.cols();

        let runners: [(String, Box<dyn FnMut()>); 1] = [(format!("gadget_product"), {
            runner(
                params.module(),
                &mut res_dft_0,
                &mut res_dft_1,
                &mut a,
                a_cols,
                &gadget_ct,
                b_cols,
                &mut tmp_bytes,
            )
        })];

        for (name, mut runner) in runners {
            let id: BenchmarkId = BenchmarkId::new(name, format!("n={}", 1 << log_n));
            b.bench_with_input(id, &(), |b: &mut criterion::Bencher<'_>, _| {
                b.iter(&mut runner)
            });
        }
    }
}

criterion_group!(benches, bench_gadget_product_inplace);
criterion_main!(benches);
