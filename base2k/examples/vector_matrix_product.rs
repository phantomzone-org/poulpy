use base2k::{
    Encoding, FFT64, MatZnxDft, MatZnxDftOps, Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps,
    ZnxInfos, ZnxLayout, alloc_aligned,
};

fn main() {
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module<FFT64> = Module::<FFT64>::new(n);
    let log_base2k: usize = 15;
    let limbs_vec: usize = 5;
    let log_k: usize = log_base2k * limbs_vec - 5;

    let rows_mat: usize = limbs_vec;
    let limbs_mat: usize = limbs_vec + 1;

    // Maximum size of the byte scratch needed
    let tmp_bytes: usize = module.vmp_prepare_tmp_bytes(rows_mat, 1, limbs_mat)
        | module.vmp_apply_dft_tmp_bytes(limbs_vec, limbs_vec, rows_mat, limbs_mat);

    let mut buf: Vec<u8> = alloc_aligned(tmp_bytes);

    let mut a_values: Vec<i64> = vec![i64::default(); n];
    a_values[1] = (1 << log_base2k) + 1;

    let mut a: VecZnx = module.new_vec_znx(1, limbs_vec);
    a.encode_vec_i64(0, log_base2k, log_k, &a_values, 32);
    a.normalize(log_base2k, &mut buf);

    a.print(n);
    println!();

    let mut mat_znx_dft: MatZnxDft<FFT64> = module.new_mat_znx_dft(rows_mat, 1, limbs_mat);

    (0..a.size()).for_each(|row_i| {
        let mut tmp: VecZnx = module.new_vec_znx(1, limbs_mat);
        tmp.at_limb_mut(row_i)[1] = 1 as i64;
        module.vmp_prepare_row(&mut mat_znx_dft, tmp.raw(), row_i, &mut buf);
    });

    let mut c_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, limbs_mat);
    module.vmp_apply_dft(&mut c_dft, &a, &mat_znx_dft, &mut buf);

    let mut c_big: VecZnxBig<FFT64> = c_dft.as_vec_znx_big();
    module.vec_znx_idft_tmp_a(&mut c_big, &mut c_dft);

    let mut res: VecZnx = module.new_vec_znx(1, limbs_vec);
    module.vec_znx_big_normalize(log_base2k, &mut res, &c_big, &mut buf);

    let mut values_res: Vec<i64> = vec![i64::default(); n];
    res.decode_vec_i64(0, log_base2k, log_k, &mut values_res);

    res.print(n);

    module.free();

    println!("{:?}", values_res)
}
