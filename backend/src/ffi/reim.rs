#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_fft_precomp {
    _unused: [u8; 0],
}
pub type REIM_FFT_PRECOMP = reim_fft_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_ifft_precomp {
    _unused: [u8; 0],
}
pub type REIM_IFFT_PRECOMP = reim_ifft_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_mul_precomp {
    _unused: [u8; 0],
}
pub type REIM_FFTVEC_MUL_PRECOMP = reim_mul_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_addmul_precomp {
    _unused: [u8; 0],
}
pub type REIM_FFTVEC_ADDMUL_PRECOMP = reim_addmul_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_from_znx32_precomp {
    _unused: [u8; 0],
}
pub type REIM_FROM_ZNX32_PRECOMP = reim_from_znx32_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_from_znx64_precomp {
    _unused: [u8; 0],
}
pub type REIM_FROM_ZNX64_PRECOMP = reim_from_znx64_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_from_tnx32_precomp {
    _unused: [u8; 0],
}
pub type REIM_FROM_TNX32_PRECOMP = reim_from_tnx32_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_to_tnx32_precomp {
    _unused: [u8; 0],
}
pub type REIM_TO_TNX32_PRECOMP = reim_to_tnx32_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_to_tnx_precomp {
    _unused: [u8; 0],
}
pub type REIM_TO_TNX_PRECOMP = reim_to_tnx_precomp;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct reim_to_znx64_precomp {
    _unused: [u8; 0],
}
pub type REIM_TO_ZNX64_PRECOMP = reim_to_znx64_precomp;
unsafe extern "C" {
    pub unsafe fn new_reim_fft_precomp(m: u32, num_buffers: u32) -> *mut REIM_FFT_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_fft_precomp_get_buffer(tables: *const REIM_FFT_PRECOMP, buffer_index: u32) -> *mut f64;
}
unsafe extern "C" {
    pub unsafe fn new_reim_fft_buffer(m: u32) -> *mut f64;
}
unsafe extern "C" {
    pub unsafe fn delete_reim_fft_buffer(buffer: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn reim_fft(tables: *const REIM_FFT_PRECOMP, data: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_ifft_precomp(m: u32, num_buffers: u32) -> *mut REIM_IFFT_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_ifft_precomp_get_buffer(tables: *const REIM_IFFT_PRECOMP, buffer_index: u32) -> *mut f64;
}
unsafe extern "C" {
    pub unsafe fn reim_ifft(tables: *const REIM_IFFT_PRECOMP, data: *mut f64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_fftvec_mul_precomp(m: u32) -> *mut REIM_FFTVEC_MUL_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_fftvec_mul(tables: *const REIM_FFTVEC_MUL_PRECOMP, r: *mut f64, a: *const f64, b: *const f64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_fftvec_addmul_precomp(m: u32) -> *mut REIM_FFTVEC_ADDMUL_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_fftvec_addmul(tables: *const REIM_FFTVEC_ADDMUL_PRECOMP, r: *mut f64, a: *const f64, b: *const f64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_from_znx32_precomp(m: u32, log2bound: u32) -> *mut REIM_FROM_ZNX32_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_from_znx32(tables: *const REIM_FROM_ZNX32_PRECOMP, r: *mut ::std::os::raw::c_void, a: *const i32);
}
unsafe extern "C" {
    pub unsafe fn reim_from_znx64(tables: *const REIM_FROM_ZNX64_PRECOMP, r: *mut ::std::os::raw::c_void, a: *const i64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_from_znx64_precomp(m: u32, maxbnd: u32) -> *mut REIM_FROM_ZNX64_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_from_znx64_simple(m: u32, log2bound: u32, r: *mut ::std::os::raw::c_void, a: *const i64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_from_tnx32_precomp(m: u32) -> *mut REIM_FROM_TNX32_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_from_tnx32(tables: *const REIM_FROM_TNX32_PRECOMP, r: *mut ::std::os::raw::c_void, a: *const i32);
}
unsafe extern "C" {
    pub unsafe fn new_reim_to_tnx32_precomp(m: u32, divisor: f64, log2overhead: u32) -> *mut REIM_TO_TNX32_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_to_tnx32(tables: *const REIM_TO_TNX32_PRECOMP, r: *mut i32, a: *const ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub unsafe fn new_reim_to_tnx_precomp(m: u32, divisor: f64, log2overhead: u32) -> *mut REIM_TO_TNX_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_to_tnx(tables: *const REIM_TO_TNX_PRECOMP, r: *mut f64, a: *const f64);
}
unsafe extern "C" {
    pub unsafe fn reim_to_tnx_simple(m: u32, divisor: f64, log2overhead: u32, r: *mut f64, a: *const f64);
}
unsafe extern "C" {
    pub unsafe fn new_reim_to_znx64_precomp(m: u32, divisor: f64, log2bound: u32) -> *mut REIM_TO_ZNX64_PRECOMP;
}
unsafe extern "C" {
    pub unsafe fn reim_to_znx64(precomp: *const REIM_TO_ZNX64_PRECOMP, r: *mut i64, a: *const ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub unsafe fn reim_to_znx64_simple(m: u32, divisor: f64, log2bound: u32, r: *mut i64, a: *const ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub unsafe fn reim_fft_simple(m: u32, data: *mut ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub unsafe fn reim_ifft_simple(m: u32, data: *mut ::std::os::raw::c_void);
}
unsafe extern "C" {
    pub unsafe fn reim_fftvec_mul_simple(
        m: u32,
        r: *mut ::std::os::raw::c_void,
        a: *const ::std::os::raw::c_void,
        b: *const ::std::os::raw::c_void,
    );
}
unsafe extern "C" {
    pub unsafe fn reim_fftvec_addmul_simple(
        m: u32,
        r: *mut ::std::os::raw::c_void,
        a: *const ::std::os::raw::c_void,
        b: *const ::std::os::raw::c_void,
    );
}
unsafe extern "C" {
    pub unsafe fn reim_from_znx32_simple(m: u32, log2bound: u32, r: *mut ::std::os::raw::c_void, x: *const i32);
}
unsafe extern "C" {
    pub unsafe fn reim_from_tnx32_simple(m: u32, r: *mut ::std::os::raw::c_void, x: *const i32);
}
unsafe extern "C" {
    pub unsafe fn reim_to_tnx32_simple(m: u32, divisor: f64, log2overhead: u32, r: *mut i32, x: *const ::std::os::raw::c_void);
}
