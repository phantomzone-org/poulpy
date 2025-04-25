use base2k::{
    BACKEND, Encoding, Infos, Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMat, VmpPMatOps,
    alloc_aligned,
};

fn main() {
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module = Module::new(n, BACKEND::FFT64);
    let log_base2k: usize = 15;
    let cols: usize = 5;
    let log_k: usize = log_base2k * cols - 5;

    let rows: usize = cols;
    let cols: usize = cols + 1;

    // Maximum size of the byte scratch needed
    let tmp_bytes: usize = module.vmp_prepare_tmp_bytes(rows, cols) | module.vmp_apply_dft_tmp_bytes(cols, cols, rows, cols);

    let mut buf: Vec<u8> = alloc_aligned(tmp_bytes);

    let mut a_values: Vec<i64> = vec![i64::default(); n];
    a_values[1] = (1 << log_base2k) + 1;

    let mut a: VecZnx = module.new_vec_znx(1, rows);
    a.encode_vec_i64(0, log_base2k, log_k, &a_values, 32);
    a.normalize(log_base2k, &mut buf);

    a.print(0, a.cols(), n);
    println!();

    let mut vmp_pmat: VmpPMat = module.new_vmp_pmat(1, rows, cols);

    (0..a.cols()).for_each(|row_i| {
        let mut tmp: VecZnx = module.new_vec_znx(1, cols);
        tmp.at_mut(row_i)[1] = 1 as i64;
        module.vmp_prepare_row(&mut vmp_pmat, tmp.raw(), row_i, &mut buf);
    });

    let mut c_dft: VecZnxDft = module.new_vec_znx_dft(1, cols);
    module.vmp_apply_dft(&mut c_dft, &a, &vmp_pmat, &mut buf);

    let mut c_big: VecZnxBig = c_dft.as_vec_znx_big();
    module.vec_znx_idft_tmp_a(&mut c_big, &mut c_dft);

    let mut res: VecZnx = module.new_vec_znx(1, rows);
    module.vec_znx_big_normalize(log_base2k, &mut res, &c_big, &mut buf);

    let mut values_res: Vec<i64> = vec![i64::default(); n];
    res.decode_vec_i64(0, log_base2k, log_k, &mut values_res);

    res.print(0, res.cols(), n);

    module.free();

    println!("{:?}", values_res)
}
