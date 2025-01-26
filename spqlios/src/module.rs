use crate::bindings::*;

pub fn create_module(N: u64, mtype: module_type_t) -> *mut MODULE {
    unsafe {
        let m = new_module_info(N, mtype);
        if m.is_null() {
            println!("Failed to create module.");
        }
        m
    }
}

#[test]
fn test_new_module_info() {
    let N: u64 = 1024;
    let module_ptr: *mut module_info_t = create_module(N, module_type_t_FFT64);
    assert!(!module_ptr.is_null());
    println!("{:?}", module_ptr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::time::Instant;
    //use test::Bencher;

    #[test]
    fn test_fft() {
        let log_bound: usize = 19;

        let n: usize = 2048;
        let m: usize = n >> 1;

        let mut a: Vec<i64> = vec![i64::default(); n];
        let mut b: Vec<i64> = vec![i64::default(); n];
        let mut c: Vec<i64> = vec![i64::default(); n];

        a.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        b[1] = 1;

        println!("{:?}", b);

        unsafe {
            let reim_fft_precomp = new_reim_fft_precomp(m as u32, 2);
            let reim_ifft_precomp = new_reim_ifft_precomp(m as u32, 1);

            let buf_a = reim_fft_precomp_get_buffer(reim_fft_precomp, 0);
            let buf_b = reim_fft_precomp_get_buffer(reim_fft_precomp, 1);
            let buf_c = reim_ifft_precomp_get_buffer(reim_ifft_precomp, 0);

            let now = Instant::now();
            (0..1024).for_each(|i| {
                reim_from_znx64_simple(
                    m as u32,
                    log_bound as u32,
                    buf_a as *mut c_void,
                    a.as_ptr(),
                );
                reim_fft(reim_fft_precomp, buf_a);

                reim_from_znx64_simple(
                    m as u32,
                    log_bound as u32,
                    buf_b as *mut c_void,
                    b.as_ptr(),
                );
                reim_fft(reim_fft_precomp, buf_b);

                reim_fftvec_mul_simple(
                    m as u32,
                    buf_c as *mut c_void,
                    buf_a as *mut c_void,
                    buf_b as *mut c_void,
                );
                reim_ifft(reim_ifft_precomp, buf_c);

                reim_to_znx64_simple(
                    m as u32,
                    m as f64,
                    log_bound as u32,
                    c.as_mut_ptr(),
                    buf_c as *mut c_void,
                )
            });

            println!("time: {}us", now.elapsed().as_micros());
            println!("{:?}", &c[..16]);
        }
    }
}
