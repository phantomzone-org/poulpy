use base2k::{Matrix3D, Module, VecZnx, VecZnxBig, VecZnxDft, VmpPMat, FFT64};

fn main() {
    let log_n = 5;
    let n = 1 << log_n;

    let module: Module = Module::new::<FFT64>(n);
    let log_base2k: usize = 15;
    let log_q: usize = 60;
    let limbs: usize = (log_q + log_base2k - 1) / log_base2k;

    let rows: usize = limbs + 1;
    let cols: usize = limbs + 1;

    // Maximum size of the byte scratch needed
    let tmp_bytes: usize = module.vmp_prepare_contiguous_tmp_bytes(rows, cols)
        | module.vmp_apply_dft_tmp_bytes(limbs, limbs, rows, cols);

    let mut buf: Vec<u8> = vec![0; tmp_bytes];

    let mut a_values: Vec<i64> = vec![i64::default(); n];
    a_values[1] = (1 << log_base2k) + 1;

    let mut a: VecZnx = module.new_vec_znx(log_base2k, log_q);
    a.from_i64(&a_values, 32);
    a.normalize(&mut buf);

    (0..a.limbs()).for_each(|i| println!("{}: {:?}", i, a.at(i)));

    let mut b_mat: Matrix3D<i64> = Matrix3D::new(rows, cols, n);

    (0..rows).for_each(|i| {
        (0..cols).for_each(|j| {
            b_mat.at_mut(i, j)[0] = (i * cols + j) as i64;
            b_mat.at_mut(i, j)[0] = (i * cols + j) as i64;
        })
    });

    //b_mat.data.iter_mut().enumerate().for_each(|(i, xi)| *xi = i as i64);

    println!();
    (0..rows).for_each(|i| {
        (0..cols).for_each(|j| println!("{} {}: {:?}", i, j, b_mat.at(i, j)));
        println!();
    });

    let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(rows, cols);
    module.vmp_prepare_contiguous(&mut vmp_pmat, &b_mat.data, &mut buf);

    (0..cols).for_each(|i| {
        (0..rows).for_each(|j| println!("{} {}: {:?}", i, j, vmp_pmat.at(i, j)));
        println!();
    });

    println!("{:?}", vmp_pmat.as_f64());

    let mut c_dft: VecZnxDft = module.new_vec_znx_dft(limbs);
    module.vmp_apply_dft(&mut c_dft, &a, &vmp_pmat, &mut buf);

    let mut c_big: VecZnxBig = c_dft.as_vec_znx_big();
    module.vec_znx_idft_tmp_a(&mut c_big, &mut c_dft, limbs);

    let mut res: VecZnx = module.new_vec_znx(log_base2k, log_q);
    module.vec_znx_big_normalize(&mut res, &c_big, &mut buf);

    let mut values_res: Vec<i64> = vec![i64::default(); n];
    res.to_i64(&mut values_res);

    (0..res.limbs()).for_each(|i| println!("{}: {:?}", i, res.at(i)));

    module.delete();
    c_dft.delete();
    vmp_pmat.delete();

    println!("{:?}", values_res)
}
