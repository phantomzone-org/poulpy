use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_glwe_decrypt(c: &mut Criterion) {
    let infos =
        GLWELayout { n: (1_u32 << 12).into(), base2k: 18_u32.into(), k: 54_u32.into(), rank: 1_u32.into() };
    poulpy_core::bench_suite::decryption::bench_glwe_decrypt::<BackendImpl, _>(&infos, c, "fft64");
}

criterion_group!(benches, bench_glwe_decrypt,);
criterion_main!(benches);
