//! CKKS Multiplication Benchmark
//!
//! Prints a formatted table comparable to the OpenFHE CKKS benchmark.
//! Does not use criterion - runs warmup + timed iterations directly.

use std::hint::black_box;
use std::time::Instant;

use poulpy_core::{
    EncryptionLayout, GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWE, GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWETensorKey, GLWETensorKeyLayout,
        GLWETensorKeyPreparedFactory, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DeviceBuf, Module, Scratch, ScratchOwned},
    source::Source,
};

const WARMUP: usize = 20;
const ITERATIONS: usize = 200;

// ── Parameters ──────────────────────────────────────────────────────────────

struct Params {
    log_n: usize,
    base2k: usize,
    k: usize,
    dsize: usize,
}

impl Params {
    const fn n(&self) -> usize {
        1 << self.log_n
    }
    fn limbs(&self) -> usize {
        self.k.div_ceil(self.base2k)
    }
    fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision(self.k as u32),
            rank: Rank(1),
        }
    }
    fn tsk_layout(&self) -> GLWETensorKeyLayout {
        let dw = self.dsize * self.base2k;
        let dnum = (self.k + dw).div_ceil(dw);
        let tsk_k = dnum * dw;
        GLWETensorKeyLayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision(tsk_k as u32),
            rank: Rank(1),
            dnum: Dnum(dnum as u32),
            dsize: Dsize(self.dsize as u32),
        }
    }
    fn tsk_enc_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        EncryptionLayout::new_from_default_sigma(self.tsk_layout()).unwrap()
    }
    fn glwe_enc_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(self.glwe_layout()).unwrap()
    }
    fn tensor_layout(&self) -> GLWELayout {
        GLWELayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision((self.k + 4 * self.base2k) as u32),
            rank: Rank(1),
        }
    }
    fn dnum(&self) -> usize {
        self.tsk_layout().dnum.0 as usize
    }
    fn log_p(&self) -> usize {
        self.tsk_layout().k.0 as usize - self.k
    }
}

const PARAMS_SQRT: &[Params] = &[
    Params {
        log_n: 12,
        base2k: 52,
        k: 52,
        dsize: 1,
    },
    Params {
        log_n: 13,
        base2k: 52,
        k: 104,
        dsize: 1,
    },
    Params {
        log_n: 14,
        base2k: 52,
        k: 260,
        dsize: 2,
    },
    Params {
        log_n: 15,
        base2k: 52,
        k: 520,
        dsize: 3,
    },
    Params {
        log_n: 16,
        base2k: 52,
        k: 1040,
        dsize: 4,
    },
];

const PARAMS_DSIZE1: &[Params] = &[
    Params {
        log_n: 12,
        base2k: 52,
        k: 52,
        dsize: 1,
    },
    Params {
        log_n: 13,
        base2k: 52,
        k: 104,
        dsize: 1,
    },
    Params {
        log_n: 14,
        base2k: 52,
        k: 260,
        dsize: 1,
    },
    Params {
        log_n: 15,
        base2k: 52,
        k: 520,
        dsize: 1,
    },
    Params {
        log_n: 16,
        base2k: 52,
        k: 1040,
        dsize: 1,
    },
];

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_tensor_key<BE: Backend>(module: &Module<BE>, p: &Params) -> poulpy_core::layouts::GLWETensorKeyPrepared<DeviceBuf<BE>, BE>
where
    Module<BE>: ModuleN + GLWESecretPreparedFactory<BE> + GLWETensorKeyEncryptSk<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let glwe_enc = p.glwe_enc_layout();
    let tsk_enc = p.tsk_enc_layout();
    let mut xs = Source::new([0u8; 32]);
    let mut xa = Source::new([1u8; 32]);
    let mut xe = Source::new([2u8; 32]);
    let mut sk = GLWESecret::alloc_from_infos(&glwe_enc);
    sk.fill_ternary_hw(192, &mut xs);
    let sz = module
        .glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_enc)
        .max(module.prepare_tensor_key_tmp_bytes(&tsk_enc));
    let mut ss: ScratchOwned<BE> = ScratchOwned::alloc(sz);
    let mut tsk = GLWETensorKey::alloc_from_infos(&tsk_enc);
    module.glwe_tensor_key_encrypt_sk(&mut tsk, &sk, &tsk_enc, &mut xa, &mut xe, ss.borrow());
    let mut tsk_p = module.alloc_tensor_key_prepared_from_infos(&tsk_enc);
    module.prepare_tensor_key(&mut tsk_p, &tsk, ss.borrow());
    tsk_p
}

/// Compute prepared tensor key size in bytes.
fn relin_key_bytes<BE: Backend>(module: &Module<BE>, p: &Params) -> usize
where
    Module<BE>: ModuleN + GLWETensorKeyPreparedFactory<BE>,
{
    let tsk = p.tsk_layout();
    module.bytes_of_tensor_key_prepared(tsk.base2k, tsk.k, tsk.rank, tsk.dnum, tsk.dsize)
}

struct Row {
    n: usize,
    limbs: usize,
    dnum: usize,
    log_q: usize,
    log_p: usize,
    log_qp: usize,
    dsize: usize,
    tensor_ms: f64,
    relin_ms: f64,
    rescale_ms: f64,
    full_ms: f64,
    relin_key_gb: f64,
}

fn bench_row<BE: Backend>(p: &Params) -> Row
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEShift<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let module = Module::<BE>::new(p.n() as u64);
    let glwe = p.glwe_layout();
    let tensor_layout = p.tensor_layout();
    let tsk_layout = p.tsk_layout();

    eprint!("  setup N={:<6}...", p.n());
    let tsk_prepared = make_tensor_key(&module, p);
    let key_bytes = relin_key_bytes(&module, p);

    let a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe);
    let b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe);
    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe);

    let tensor_bytes = poulpy_core::layouts::GLWETensor::bytes_of_from_infos(&tensor_layout);
    let op_bytes = module
        .glwe_tensor_apply_tmp_bytes(&tensor_layout, &glwe, &glwe)
        .max(module.glwe_tensor_relinearize_tmp_bytes(&glwe, &tensor_layout, &tsk_layout))
        .max(module.glwe_shift_tmp_bytes());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tensor_bytes + op_bytes);
    let tsk_size = tsk_prepared.size();
    let base2k = p.base2k;

    // Precompute tensor for relin benchmark
    {
        let s = scratch.borrow();
        let (mut tensor, s_rest) = s.take_glwe_tensor(&tensor_layout);
        module.glwe_tensor_apply(0, &mut tensor, &a, a.max_k().as_usize(), &b, b.max_k().as_usize(), s_rest);
    }

    // ── tensor ──
    let tensor_ms = {
        for _ in 0..WARMUP {
            let s = scratch.borrow();
            let (mut t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_apply(0, &mut t, &a, a.max_k().as_usize(), &b, b.max_k().as_usize(), sr);
            black_box(());
        }
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let s = scratch.borrow();
            let (mut t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_apply(0, &mut t, &a, a.max_k().as_usize(), &b, b.max_k().as_usize(), sr);
            black_box(());
        }
        start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64
    };

    // ── relin ──
    let relin_ms = {
        for _ in 0..WARMUP {
            let s = scratch.borrow();
            let (t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_relinearize(&mut res, &t, &tsk_prepared, tsk_size, sr);
            black_box(());
        }
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let s = scratch.borrow();
            let (t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_relinearize(&mut res, &t, &tsk_prepared, tsk_size, sr);
            black_box(());
        }
        start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64
    };

    // ── rescale ──
    let rescale_ms = {
        for _ in 0..WARMUP {
            module.glwe_lsh_inplace(&mut res, base2k, scratch.borrow());
            black_box(());
        }
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            module.glwe_lsh_inplace(&mut res, base2k, scratch.borrow());
            black_box(());
        }
        start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64
    };

    // ── full mul ──
    let full_ms = {
        for _ in 0..WARMUP {
            let s = scratch.borrow();
            let (mut t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_apply(0, &mut t, &a, a.max_k().as_usize(), &b, b.max_k().as_usize(), sr);
            module.glwe_tensor_relinearize(&mut res, &t, &tsk_prepared, tsk_size, sr);
            module.glwe_lsh_inplace(&mut res, base2k, sr);
            black_box(());
        }
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let s = scratch.borrow();
            let (mut t, sr) = s.take_glwe_tensor(&tensor_layout);
            module.glwe_tensor_apply(0, &mut t, &a, a.max_k().as_usize(), &b, b.max_k().as_usize(), sr);
            module.glwe_tensor_relinearize(&mut res, &t, &tsk_prepared, tsk_size, sr);
            module.glwe_lsh_inplace(&mut res, base2k, sr);
            black_box(());
        }
        start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64
    };

    eprintln!(" done");

    Row {
        n: p.n(),
        limbs: p.limbs(),
        dnum: p.dnum(),
        log_q: p.k,
        log_p: p.log_p(),
        log_qp: p.k + p.log_p(),
        dsize: p.dsize,
        tensor_ms,
        relin_ms,
        rescale_ms,
        full_ms,
        relin_key_gb: key_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    }
}

fn print_table(title: &str, rows: &[Row]) {
    println!();
    println!("{title}");
    println!("Warmup: {WARMUP} | Iterations: {ITERATIONS}");
    println!();
    println!(
        "{:>7} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "N",
        "limbs",
        "dsize",
        "dnum",
        "logQ",
        "logP",
        "logQP",
        "tensor(ms)",
        "relin(ms)",
        "rescale(ms)",
        "full(ms)",
        "relinK(GB)"
    );
    println!("{}", "-".repeat(105));
    for r in rows {
        println!(
            "{:>7} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>11.3} {:>11.3} {:>11.3} {:>11.3} {:>11.4}",
            r.n,
            r.limbs,
            r.dsize,
            r.dnum,
            r.log_q,
            r.log_p,
            r.log_qp,
            r.tensor_ms,
            r.relin_ms,
            r.rescale_ms,
            r.full_ms,
            r.relin_key_gb
        );
    }
}

fn run_table<BE: Backend>(label: &str)
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWETensoring<BE>
        + GLWEShift<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    eprintln!("Running dsize=1 ({label})...");
    let rows_d1: Vec<Row> = PARAMS_DSIZE1.iter().map(bench_row::<BE>).collect();
    print_table(
        &format!("CKKS Multiplication Benchmark [{label}] — dsize=1 (HYBRID-like)"),
        &rows_d1,
    );

    eprintln!("Running sqrt ({label})...");
    let rows_sqrt: Vec<Row> = PARAMS_SQRT.iter().map(bench_row::<BE>).collect();
    print_table(
        &format!("CKKS Multiplication Benchmark [{label}] — sqrt (balanced)"),
        &rows_sqrt,
    );
}

fn main() {
    // Select backends via POULPY_BENCH_BACKENDS env var (comma-separated).
    // Default: only the fastest compiled-in backend.
    // Available: ntt120-ref, ntt-ifma-ref, ntt120-avx, ntt-ifma, all
    let backends = std::env::var("POULPY_BENCH_BACKENDS").unwrap_or_default();
    let backends: Vec<&str> = if backends.is_empty() {
        // Default: pick the fastest available
        #[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
        {
            vec!["ntt-ifma"]
        }
        #[cfg(all(not(feature = "enable-ifma"), feature = "enable-avx", target_arch = "x86_64"))]
        {
            vec!["ntt120-avx"]
        }
        #[cfg(not(any(
            all(feature = "enable-ifma", target_arch = "x86_64"),
            all(feature = "enable-avx", target_arch = "x86_64")
        )))]
        {
            vec!["ntt120-ref"]
        }
    } else {
        backends.split(',').map(str::trim).collect()
    };

    let run_all = backends.contains(&"all");

    if run_all || backends.contains(&"ntt120-ref") {
        run_table::<poulpy_cpu_ref::NTT120Ref>("ntt120-ref");
    }
    if run_all || backends.contains(&"ntt-ifma-ref") {
        run_table::<poulpy_cpu_ref::NTTIfmaRef>("ntt-ifma-ref");
    }
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    if run_all || backends.contains(&"ntt120-avx") {
        run_table::<poulpy_cpu_avx::NTT120Avx>("ntt120-avx");
    }
    #[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
    if run_all || backends.contains(&"ntt-ifma") {
        run_table::<poulpy_cpu_ifma::NTTIfma>("ntt-ifma");
    }
}
