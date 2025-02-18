use base2k::{FFT64, Module, SvpPPolOps, VecZnx, VmpPMat, alloc_aligned_u8};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rlwe::{
    ciphertext::{Ciphertext, new_gadget_ciphertext},
    elem::Elem,
    encryptor::{encrypt_grlwe_sk_thread_safe, encrypt_grlwe_sk_tmp_bytes},
    evaluator::{gadget_product_inplace_thread_safe, gadget_product_tmp_bytes},
    key_generator::gen_switching_key_thread_safe_tmp_bytes,
    keys::SecretKey,
    parameters::{Parameters, ParametersLiteral},
};
use sampling::source::Source;

fn gadget_product_inplace(c: &mut Criterion) {
    fn gadget_product<'a>(
        module: &'a Module,
        elem: &'a mut Elem<VecZnx>,
        gadget_ct: &'a Ciphertext<VmpPMat>,
        tmp_bytes: &'a mut [u8],
    ) -> Box<dyn FnMut() + 'a> {
        Box::new(move || {
            gadget_product_inplace_thread_safe::<true, _>(module, elem, gadget_ct, tmp_bytes)
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
            params.decrypt_rlwe_thread_safe_tmp_byte(params.log_q())
                | params.encrypt_rlwe_sk_tmp_bytes(params.log_q())
                | gen_switching_key_thread_safe_tmp_bytes(
                    params.module(),
                    params.log_base2k(),
                    params.limbs_q(),
                    params.log_q(),
                )
                | gadget_product_tmp_bytes(
                    params.module(),
                    params.log_base2k(),
                    params.log_q(),
                    params.log_q(),
                    params.limbs_q(),
                    params.log_qp(),
                )
                | encrypt_grlwe_sk_tmp_bytes(
                    params.module(),
                    params.log_base2k(),
                    params.limbs_qp(),
                    params.log_qp(),
                ),
            64,
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
            params.limbs_q(),
            params.log_qp(),
        );

        encrypt_grlwe_sk_thread_safe(
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

        let runners: [(String, Box<dyn FnMut()>); 1] = [(format!("gadget_product"), {
            gadget_product(params.module(), &mut ct.0, &gadget_ct, &mut tmp_bytes)
        })];

        for (name, mut runner) in runners {
            let id: BenchmarkId = BenchmarkId::new(name, format!("n={}", 1 << log_n));
            b.bench_with_input(id, &(), |b: &mut criterion::Bencher<'_>, _| {
                b.iter(&mut runner)
            });
        }
    }
}

criterion_group!(benches, gadget_product_inplace);
criterion_main!(benches);
