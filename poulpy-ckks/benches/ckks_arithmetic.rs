use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_ckks::{
    encoding::classical::{encode, encode_tmp_bytes},
    layouts::{
        ciphertext::{CKKSCiphertext, CKKSCiphertextToRef},
        plaintext::CKKSPlaintext,
    },
    leveled::{
        arithmetic::{
            add_const_ct, add_const_ct_inplace, add_ct_ct, add_ct_ct_inplace, add_pt_ct, add_pt_ct_inplace, neg_ct,
            neg_ct_inplace, sub_const_ct, sub_const_ct_inplace, sub_ct_ct, sub_ct_ct_inplace, sub_pt_ct, sub_pt_ct_inplace,
        },
        encryption::{encrypt_sk, encrypt_sk_tmp_bytes},
    },
};
use poulpy_core::{
    GLWEAdd, GLWEEncryptSk, GLWESub, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecret, GLWESecretPreparedFactory, Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAlloc, VecZnxIdftApplyConsume, VecZnxNegate, VecZnxNegateInplace,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned, ToOwnedDeep, ZnxView, ZnxViewMut},
    source::Source,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type FFTBackendImpl = poulpy_cpu_avx::FFT64Avx;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type FFTBackendImpl = poulpy_cpu_ref::FFT64Ref;

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type NTTBackendImpl = poulpy_cpu_avx::NTT120Avx;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type NTTBackendImpl = poulpy_cpu_ref::NTT120Ref;

#[derive(Clone, Copy)]
struct Params {
    label: &'static str,
    n: usize,
    hw: u32,
    base2k: u32,
    log_delta: u32,
    k: u32,
}

const FFT64_PARAMS: Params = Params {
    label: "fft64",
    n: 65536,
    hw: 256,
    base2k: 19,
    log_delta: 40,
    k: 81 * 19, // 1539 bits
};

const NTT120_PARAMS: Params = Params {
    label: "ntt120",
    n: 65536,
    hw: 256,
    base2k: 52,
    log_delta: 40,
    k: 29 * 52, // 1508 bits
};

struct BenchSetup<BE: Backend> {
    module: Module<BE>,
    scratch_be: ScratchOwned<BE>,
    ct_a: CKKSCiphertext<Vec<u8>>,
    ct_b: CKKSCiphertext<Vec<u8>>,
    pt_b: CKKSPlaintext<Vec<u8>>,
    const_re: f64,
    const_im: f64,
}

fn random_slots(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let m = n / 2;
    let re = (0..m).map(|_| rng.random_range(-0.5..0.5)).collect();
    let im = (0..m).map(|_| rng.random_range(-0.5..0.5)).collect();
    (re, im)
}

fn setup_benchmark<BE, CE>(params: Params) -> BenchSetup<BE>
where
    BE: Backend,
    CE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    Module<BE>: ModuleNew<BE> + ModuleN + GLWESecretPreparedFactory<BE> + GLWEEncryptSk<BE>,
    Module<CE>: ModuleNew<CE>
        + ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let module = Module::<BE>::new(params.n as u64);
    let codec = Module::<CE>::new(params.n as u64);
    let n = Degree(params.n as u32);
    let base2k = Base2K(params.base2k);
    let k = TorusPrecision(params.k);
    let glwe_layout = GLWELayout {
        n,
        base2k,
        k,
        rank: Rank(1),
    };

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe_layout);
    sk.fill_ternary_hw(params.hw as usize, &mut source_xs);
    let mut sk_prepared = GLWESecretPrepared::alloc_from_infos(&module, &glwe_layout);
    sk_prepared.prepare(&module, &sk);

    let mut pt_a = CKKSPlaintext::alloc(n, base2k, k, params.log_delta);
    let mut pt_b = CKKSPlaintext::alloc(n, base2k, k, params.log_delta);
    let mut ct_a = CKKSCiphertext::alloc(n, base2k, k, params.log_delta);
    let mut ct_b = CKKSCiphertext::alloc(n, base2k, k, params.log_delta);

    let mut scratch_be = ScratchOwned::<BE>::alloc(encrypt_sk_tmp_bytes(&module, &ct_a));
    let mut scratch_ce = ScratchOwned::<CE>::alloc(encode_tmp_bytes(&codec));

    let (re_a, im_a) = random_slots(params.n, 0xA5A5_0001);
    let (re_b, im_b) = random_slots(params.n, 0x5A5A_0002);
    encode(&codec, &mut pt_a, &re_a, &im_a, scratch_ce.borrow());
    encode(&codec, &mut pt_b, &re_b, &im_b, scratch_ce.borrow());
    encrypt_sk(
        &module,
        &mut ct_a,
        &pt_a,
        &sk_prepared,
        &mut source_xa,
        &mut source_xe,
        scratch_be.borrow(),
    );
    encrypt_sk(
        &module,
        &mut ct_b,
        &pt_b,
        &sk_prepared,
        &mut Source::new([3u8; 32]),
        &mut Source::new([4u8; 32]),
        scratch_be.borrow(),
    );

    BenchSetup {
        module,
        scratch_be,
        ct_a,
        ct_b,
        pt_b,
        const_re: 0.125,
        const_im: -0.25,
    }
}

fn reset_ct(dst: &mut CKKSCiphertext<Vec<u8>>, src: &CKKSCiphertext<Vec<u8>>) {
    dst.log_delta = src.log_delta;
    dst.inner.data_mut().raw_mut().copy_from_slice(src.inner.data().raw());
}

fn bench_suite<BE, CE>(c: &mut Criterion, params: Params)
where
    BE: Backend,
    CE: Backend<ScalarPrep = f64, ScalarBig = i64>,
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd
        + GLWESub
        + VecZnxNegate
        + VecZnxNegateInplace,
    Module<CE>: ModuleNew<CE>
        + ModuleN
        + VecZnxDftAlloc<CE>
        + VecZnxIdftApplyConsume<CE>
        + VecZnxBigNormalize<CE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<CE>: ScratchOwnedAlloc<CE> + ScratchOwnedBorrow<CE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut group = c.benchmark_group(params.label);
    let BenchSetup {
        module,
        mut scratch_be,
        ct_a,
        ct_b,
        pt_b,
        const_re,
        const_im,
    } = setup_benchmark::<BE, CE>(params);

    let mut ct_out = CKKSCiphertext {
        inner: ct_a.to_ref().inner.to_owned_deep(),
        log_delta: ct_a.log_delta,
    };
    let mut ct_work = CKKSCiphertext {
        inner: ct_a.to_ref().inner.to_owned_deep(),
        log_delta: ct_a.log_delta,
    };

    group.bench_with_input(BenchmarkId::from_parameter("add_ct_ct"), &(), |b, _| {
        b.iter(|| {
            add_ct_ct(&module, &mut ct_out, black_box(&ct_a), black_box(&ct_b));
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("add_ct_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_ct_ct_inplace(&module, &mut ct_work, black_box(&ct_b));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("add_pt_ct"), &(), |b, _| {
        b.iter(|| {
            add_pt_ct(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b));
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("add_pt_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_pt_ct_inplace(&module, &mut ct_work, black_box(&pt_b));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("add_const_ct"), &(), |b, _| {
        b.iter(|| {
            add_const_ct(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(const_re),
                black_box(const_im),
            );
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("add_const_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_const_ct_inplace(&module, &mut ct_work, black_box(const_re), black_box(const_im));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_ct_ct"), &(), |b, _| {
        b.iter(|| {
            sub_ct_ct(&module, &mut ct_out, black_box(&ct_a), black_box(&ct_b));
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_ct_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_ct_ct_inplace(&module, &mut ct_work, black_box(&ct_b));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_pt_ct"), &(), |b, _| {
        b.iter(|| {
            sub_pt_ct(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b));
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_pt_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_pt_ct_inplace(&module, &mut ct_work, black_box(&pt_b));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_const_ct"), &(), |b, _| {
        b.iter(|| {
            sub_const_ct(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(const_re),
                black_box(const_im),
            );
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("sub_const_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_const_ct_inplace(&module, &mut ct_work, black_box(const_re), black_box(const_im));
            black_box(&ct_work);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("neg_ct"), &(), |b, _| {
        b.iter(|| {
            neg_ct(&module, &mut ct_out, black_box(&ct_a));
            black_box(&ct_out);
        })
    });

    group.bench_with_input(BenchmarkId::from_parameter("neg_ct_inplace"), &(), |b, _| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            neg_ct_inplace(&module, &mut ct_work);
            black_box(&ct_work);
        })
    });

    black_box(scratch_be.borrow());
    group.finish();
}

fn fft64(c: &mut Criterion) {
    bench_suite::<FFTBackendImpl, FFTBackendImpl>(c, FFT64_PARAMS);
}

fn ntt120(c: &mut Criterion) {
    bench_suite::<NTTBackendImpl, FFTBackendImpl>(c, NTT120_PARAMS);
}

fn criterion_config() -> Criterion {
    Criterion::default().measurement_time(Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = fft64, ntt120
}
criterion_main!(benches);
