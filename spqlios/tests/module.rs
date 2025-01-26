use spqlios::bindings::{module_info_t, module_type_t_FFT64};
use spqlios::module::create_module;

#[test]
fn test_new_module_info() {
    let N: u64 = 1024;
    let module_ptr: *mut module_info_t = create_module(N, module_type_t_FFT64);
    assert!(!module_ptr.is_null());
}
