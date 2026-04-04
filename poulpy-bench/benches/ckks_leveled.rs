use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_ckks::{
    encoding::classical::encode,
    layouts::{ciphertext::CKKSCiphertext, plaintext::CKKSPlaintext, plaintext_prepared::CKKSPlaintextPrepared},
    leveled::{
        encryption::{encrypt_sk, encrypt_sk_tmp_bytes},
        operations::{
            add::{
                add, add_const, add_const_inplace, add_const_tmp_bytes, add_inplace, add_prepared_pt, add_prepared_pt_inplace,
                add_pt, add_pt_inplace, add_pt_tmp_bytes,
            },
            conjugate::{conjugate, conjugate_inplace, conjugate_tmp_bytes},
            level::{div_pow2, div_pow2_inplace, div_pow2_tmp_bytes},
            mul::{
                mul, mul_const, mul_const_inplace, mul_const_tmp_bytes, mul_int, mul_int_inplace, mul_int_inplace_tmp_bytes,
                mul_prepared_pt, mul_prepared_pt_inplace, mul_pt, mul_pt_inplace, mul_pt_tmp_bytes, mul_tmp_bytes,
            },
            neg::{neg, neg_inplace},
            rotate::{rotate, rotate_inplace, rotate_tmp_bytes},
            sub::{
                sub, sub_const, sub_const_inplace, sub_const_tmp_bytes, sub_inplace, sub_prepared_pt, sub_prepared_pt_inplace,
                sub_pt, sub_pt_inplace, sub_pt_tmp_bytes,
            },
        },
    },
};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared,
    GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESecret, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared,
    LWEInfos, Rank, SetGLWEInfos, TorusPrecision, prepared::GLWESecretPrepared,
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{GaloisElement, Module, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[cfg(all(feature = "enable-avx", feature = "enable-ifma", not(clippy)))]
compile_error!(
    "ckks_leveled benchmark backend features are mutually exclusive; enable only one of `enable-avx` or `enable-ifma`."
);

#[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
type BE = poulpy_cpu_ifma::NTTIfma;
#[cfg(all(feature = "enable-avx", not(feature = "enable-ifma"), target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::NTT120Avx;
#[cfg(not(any(
    all(feature = "enable-ifma", target_arch = "x86_64"),
    all(feature = "enable-avx", not(feature = "enable-ifma"), target_arch = "x86_64")
)))]
type BE = poulpy_cpu_ref::NTT120Ref;

const N: usize = 1 << 15;
const BASE2K: u32 = 52;
const K: u32 = 728;
const LOG_DELTA: u32 = 40;
const HW: usize = 192;
const DSIZE: u32 = 1;

struct Setup {
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    ct_a: CKKSCiphertext<Vec<u8>>,
    ct_b: CKKSCiphertext<Vec<u8>>,
    pt_b: CKKSPlaintext<Vec<u8>>,
    pt_b_prepared: CKKSPlaintextPrepared<Vec<u8>>,
    tsk: GLWETensorKeyPrepared<Vec<u8>, BE>,
    rot_key: GLWEAutomorphismKeyPrepared<Vec<u8>, BE>,
    conj_key: GLWEAutomorphismKeyPrepared<Vec<u8>, BE>,
}

fn setup() -> Setup {
    let module = Module::<BE>::new(N as u64);
    let degree = Degree(N as u32);
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(K);
    let glwe = GLWELayout {
        n: degree,
        base2k,
        k,
        rank: Rank(1),
    };
    let dnum = K.div_ceil(DSIZE * BASE2K);
    let tsk_layout = GLWETensorKeyLayout {
        n: degree,
        base2k,
        k: TorusPrecision(K + DSIZE * BASE2K),
        rank: Rank(1),
        dsize: Dsize(DSIZE),
        dnum: Dnum(dnum),
    };

    let mut xs = Source::new([0u8; 32]);
    let mut xa = Source::new([1u8; 32]);
    let mut xe = Source::new([2u8; 32]);

    let mut sk = GLWESecret::alloc_from_infos(&glwe);
    sk.fill_ternary_hw(HW, &mut xs);
    let mut sk_prep = GLWESecretPrepared::alloc_from_infos(&module, &glwe);
    sk_prep.prepare(&module, &sk);

    let mut tsk_raw = GLWETensorKey::alloc_from_infos(&tsk_layout);
    let mut scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&module, &CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA))
            .max(GLWETensorKeyPrepared::prepare_tmp_bytes(&module, &tsk_layout))
            .max(GLWETensorKey::encrypt_sk_tmp_bytes(&module, &tsk_layout)),
    );
    tsk_raw.encrypt_sk(&module, &sk, &mut xa, &mut xe, scratch.borrow());
    let mut tsk = GLWETensorKeyPrepared::alloc_from_infos(&module, &tsk_layout);
    tsk.prepare(&module, &tsk_raw, scratch.borrow());

    let atk_dnum = K.div_ceil(DSIZE * BASE2K);
    let atk_layout = GLWEAutomorphismKeyLayout {
        n: degree,
        base2k,
        k: TorusPrecision(K + DSIZE * BASE2K),
        rank: Rank(1),
        dsize: Dsize(DSIZE),
        dnum: Dnum(atk_dnum),
    };
    scratch = ScratchOwned::<BE>::alloc(
        GLWEAutomorphismKey::encrypt_sk_tmp_bytes(&module, &atk_layout)
            .max(module.prepare_glwe_automorphism_key_tmp_bytes(&atk_layout)),
    );

    let rot_gal = module.galois_element(1);
    let mut rot_raw = GLWEAutomorphismKey::alloc_from_infos(&atk_layout);
    rot_raw.encrypt_sk(&module, rot_gal, &sk, &mut xa, &mut xe, scratch.borrow());
    let mut rot_key = GLWEAutomorphismKeyPrepared::alloc_from_infos(&module, &atk_layout);
    rot_key.prepare(&module, &rot_raw, scratch.borrow());

    let mut conj_raw = GLWEAutomorphismKey::alloc_from_infos(&atk_layout);
    conj_raw.encrypt_sk(&module, -1, &sk, &mut xa, &mut xe, scratch.borrow());
    let mut conj_key = GLWEAutomorphismKeyPrepared::alloc_from_infos(&module, &atk_layout);
    conj_key.prepare(&module, &conj_raw, scratch.borrow());

    let ct_tmp = CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA);
    scratch = ScratchOwned::<BE>::alloc(
        encrypt_sk_tmp_bytes(&module, &ct_tmp)
            .max(add_pt_tmp_bytes(&module, &ct_tmp))
            .max(add_const_tmp_bytes(&module, &ct_tmp))
            .max(sub_pt_tmp_bytes(&module, &ct_tmp))
            .max(sub_const_tmp_bytes(&module, &ct_tmp))
            .max(mul_tmp_bytes(&module, &ct_tmp, &ct_tmp, &tsk))
            .max(mul_pt_tmp_bytes(&module, &ct_tmp))
            .max(mul_const_tmp_bytes(&module, &ct_tmp))
            .max(mul_int_inplace_tmp_bytes(&module, &ct_tmp, 1))
            .max(div_pow2_tmp_bytes(&module))
            .max(rotate_tmp_bytes(&module, &ct_tmp, &rot_key))
            .max(conjugate_tmp_bytes(&module, &ct_tmp, &conj_key)),
    );

    let mut rng = StdRng::seed_from_u64(0xC115);
    let m = N / 2;
    let re_a: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
    let im_a: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
    let re_b: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
    let im_b: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut pt_a = CKKSPlaintext::alloc(degree, base2k, LOG_DELTA);
    let mut pt_b = CKKSPlaintext::alloc(degree, base2k, LOG_DELTA);
    encode(&mut pt_a, &re_a, &im_a);
    encode(&mut pt_b, &re_b, &im_b);

    let mut ct_a = CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA);
    let mut ct_b = CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA);
    encrypt_sk(&module, &mut ct_a, &pt_a, &sk_prep, &mut xa, &mut xe, scratch.borrow());
    encrypt_sk(&module, &mut ct_b, &pt_b, &sk_prep, &mut xa, &mut xe, scratch.borrow());

    let ct_size = k.0.div_ceil(base2k.0) as usize;
    let max_k = TorusPrecision(base2k.0 * ct_size as u32);
    let pt_b_prepared = CKKSPlaintextPrepared::alloc_and_prepare(&module, degree, base2k, max_k, &pt_b, scratch.borrow());

    Setup {
        module,
        scratch,
        ct_a,
        ct_b,
        pt_b,
        pt_b_prepared,
        tsk,
        rot_key,
        conj_key,
    }
}

fn reset_ct(dst: &mut CKKSCiphertext<Vec<u8>>, src: &CKKSCiphertext<Vec<u8>>) {
    dst.log_delta = src.log_delta;
    dst.inner.set_k(src.inner.k());
    dst.inner.data_mut().size = src.inner.size();
    dst.inner.data_mut().raw_mut().copy_from_slice(src.inner.data().raw());
}

fn bench_ckks_leveled(c: &mut Criterion) {
    let Setup {
        module,
        mut scratch,
        ct_a,
        ct_b,
        pt_b,
        pt_b_prepared,
        tsk,
        rot_key,
        conj_key,
    } = setup();

    let degree = Degree(N as u32);
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(K);
    let mut ct_out = CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA);
    let mut ct_work = CKKSCiphertext::alloc(degree, base2k, k, LOG_DELTA);

    reset_ct(&mut ct_out, &ct_a);
    reset_ct(&mut ct_work, &ct_a);

    let mut group = c.benchmark_group("ckks_leveled");

    group.bench_function("add", |b| {
        b.iter(|| add(&module, &mut ct_out, black_box(&ct_a), black_box(&ct_b)))
    });
    group.bench_function("add_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_inplace(&module, &mut ct_work, black_box(&ct_b))
        })
    });
    group.bench_function("add_pt", |b| {
        b.iter(|| add_pt(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b), scratch.borrow()))
    });
    group.bench_function("add_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_pt_inplace(&module, &mut ct_work, black_box(&pt_b), scratch.borrow())
        })
    });
    group.bench_function("add_prepared_pt", |b| {
        b.iter(|| add_prepared_pt(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b_prepared)))
    });
    group.bench_function("add_prepared_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_prepared_pt_inplace(&module, &mut ct_work, black_box(&pt_b_prepared))
        })
    });
    group.bench_function("add_const", |b| {
        b.iter(|| {
            add_const(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(0.5),
                black_box(-0.3),
                scratch.borrow(),
            )
        })
    });
    group.bench_function("add_const_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            add_const_inplace(&module, &mut ct_work, black_box(0.5), black_box(-0.3), scratch.borrow())
        })
    });

    group.bench_function("sub", |b| {
        b.iter(|| sub(&module, &mut ct_out, black_box(&ct_a), black_box(&ct_b)))
    });
    group.bench_function("sub_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_inplace(&module, &mut ct_work, black_box(&ct_b))
        })
    });
    group.bench_function("sub_pt", |b| {
        b.iter(|| sub_pt(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b), scratch.borrow()))
    });
    group.bench_function("sub_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_pt_inplace(&module, &mut ct_work, black_box(&pt_b), scratch.borrow())
        })
    });
    group.bench_function("sub_prepared_pt", |b| {
        b.iter(|| sub_prepared_pt(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b_prepared)))
    });
    group.bench_function("sub_prepared_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_prepared_pt_inplace(&module, &mut ct_work, black_box(&pt_b_prepared))
        })
    });
    group.bench_function("sub_const", |b| {
        b.iter(|| {
            sub_const(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(0.5),
                black_box(-0.3),
                scratch.borrow(),
            )
        })
    });
    group.bench_function("sub_const_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            sub_const_inplace(&module, &mut ct_work, black_box(0.5), black_box(-0.3), scratch.borrow())
        })
    });

    group.bench_function("neg", |b| b.iter(|| neg(&module, &mut ct_out, black_box(&ct_a))));
    group.bench_function("neg_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            neg_inplace(&module, &mut ct_work)
        })
    });

    group.bench_function("mul", |b| {
        b.iter(|| {
            reset_ct(&mut ct_out, &ct_a);
            mul(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(&ct_b),
                &tsk,
                scratch.borrow(),
            );
        })
    });
    group.bench_function("mul_pt", |b| {
        b.iter(|| {
            reset_ct(&mut ct_out, &ct_a);
            mul_pt(&module, &mut ct_out, black_box(&ct_a), black_box(&pt_b), scratch.borrow());
        })
    });
    group.bench_function("mul_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            mul_pt_inplace(&module, &mut ct_work, black_box(&pt_b), scratch.borrow());
        })
    });
    group.bench_function("mul_prepared_pt", |b| {
        b.iter(|| {
            reset_ct(&mut ct_out, &ct_a);
            mul_prepared_pt(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(&pt_b_prepared),
                scratch.borrow(),
            );
        })
    });
    group.bench_function("mul_prepared_pt_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            mul_prepared_pt_inplace(&module, &mut ct_work, black_box(&pt_b_prepared), scratch.borrow());
        })
    });
    group.bench_function("mul_const", |b| {
        b.iter(|| {
            reset_ct(&mut ct_out, &ct_a);
            mul_const(
                &module,
                &mut ct_out,
                black_box(&ct_a),
                black_box(0.5),
                black_box(-0.3),
                scratch.borrow(),
            );
        })
    });
    group.bench_function("mul_const_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            mul_const_inplace(&module, &mut ct_work, black_box(0.5), black_box(-0.3), scratch.borrow());
        })
    });
    group.bench_function("mul_int", |b| {
        b.iter(|| {
            reset_ct(&mut ct_out, &ct_a);
            mul_int(&module, &mut ct_out, black_box(&ct_a), black_box(3), scratch.borrow())
        })
    });
    group.bench_function("mul_int_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            mul_int_inplace(&module, &mut ct_work, black_box(3), scratch.borrow());
        })
    });

    group.bench_function("div_pow2", |b| {
        b.iter(|| div_pow2(&module, &mut ct_out, black_box(&ct_a), black_box(1usize), scratch.borrow()))
    });
    group.bench_function("div_pow2_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            div_pow2_inplace(&module, &mut ct_work, black_box(1usize), scratch.borrow());
        })
    });

    group.bench_function("rotate", |b| {
        b.iter(|| rotate(&module, &mut ct_out, black_box(&ct_a), &rot_key, scratch.borrow()))
    });
    group.bench_function("rotate_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            rotate_inplace(&module, &mut ct_work, &rot_key, scratch.borrow());
        })
    });
    group.bench_function("conjugate", |b| {
        b.iter(|| conjugate(&module, &mut ct_out, black_box(&ct_a), &conj_key, scratch.borrow()))
    });
    group.bench_function("conjugate_inplace", |b| {
        b.iter(|| {
            reset_ct(&mut ct_work, &ct_a);
            conjugate_inplace(&module, &mut ct_work, &conj_key, scratch.borrow());
        })
    });

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().measurement_time(Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_ckks_leveled
}
criterion_main!(benches);
