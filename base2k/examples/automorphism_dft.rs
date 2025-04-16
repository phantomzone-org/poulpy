use base2k::{
    alloc_aligned, Module, VecZnx, VecZnxBig, VecZnxDft, VecZnxDftOps, VecZnxOps, VecZnxBigOps, BACKEND
};


fn main() {
    let module: Module = Module::new(32, BACKEND::FFT64);
    let cols: usize = 1;
    let log_base2k: usize = 21;
    let mut a: VecZnx = module.new_vec_znx(cols);
    let mut a_dft: VecZnxDft = module.new_vec_znx_dft(cols);
    let mut a_big: VecZnxBig = module.new_vec_znx_big(cols);

    let mut b_dft: VecZnxDft = module.new_vec_znx_dft(cols);

    let scale: f64 = (1<<15) as f64;

    let data: &mut [f64] = a_dft.at_mut::<f64>(&module, 0);
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = (i as f64)*scale
    });

    let data: &mut [f64] = b_dft.at_mut::<f64>(&module, 0);
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = (i as f64)*scale
    });

    let gal_el: i64 = -5;

    let mut tmp_bytes: Vec<u8> = alloc_aligned(module.vec_znx_idft_tmp_bytes() | module.vec_znx_normalize_tmp_bytes());

    module.vec_znx_idft(&mut a_big, &a_dft, cols, &mut tmp_bytes);
    module.vec_znx_big_automorphism_inplace(gal_el, &mut a_big);
    module.vec_znx_big_normalize(log_base2k, &mut a, &a_big, &mut tmp_bytes);
    println!("a: {:?}", a.at(0));
    module.vec_znx_dft(&mut a_dft, &a, cols);
    
    let a_f64: &[f64] = a_dft.at(&module, 0);
    a_f64.iter().for_each(|x| {
        print!("{} ", (*x/scale).round() as i64);
    });
    println!();

    module.vec_znx_dft_automorphism_inplace(gal_el, &mut b_dft, cols, &mut tmp_bytes);

    let b_f64: &[f64] = b_dft.at(&module, 0);
    b_f64.iter().for_each(|x| {
        print!("{} ", (*x/scale).round() as i64);
    });
    println!();

    module.free();
}

