pub mod test_suite;

#[cfg(test)]
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
mod fft64_ref;

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod fft64_avx;
