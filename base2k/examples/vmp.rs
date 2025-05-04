// use base2k::{
//     Encoding, FFT64, MatZnxDft, MatZnxDftOps, Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps,
//     ZnxInfos, ZnxLayout, alloc_aligned,
// };

fn main() {
    // let log_n: i32 = 5;
    // let n: usize = 1 << log_n;

    // let module: Module<FFT64> = Module::<FFT64>::new(n);
    // let log_base2k: usize = 15;

    // let a_cols: usize = 2;
    // let a_size: usize = 5;

    // let log_k: usize = log_base2k * a_size - 5;

    // let mat_rows: usize = a_size;
    // let mat_cols_in: usize = a_cols;
    // let mat_cols_out: usize = 2;
    // let mat_size: usize = a_size + 1;

    // let mut tmp_bytes_vmp: Vec<u8> = alloc_aligned(
    //     module.vmp_prepare_row_tmp_bytes(mat_cols_out, mat_size)
    //         | module.vmp_apply_dft_tmp_bytes(
    //             a_size,
    //             a_size,
    //             mat_rows,
    //             mat_cols_in,
    //             mat_cols_out,
    //             mat_size,
    //         ),
    // );

    // let mut tmp_bytes_dft: Vec<u8> = alloc_aligned(module.bytes_of_vec_znx_dft(mat_cols_out, mat_size));

    // let mut a: VecZnx = module.new_vec_znx(a_cols, a_size);

    // (0..a_cols).for_each(|i| {
    //     let mut values: Vec<i64> = vec![i64::default(); n];
    //     values[1 + i] = (1 << log_base2k) + 1;
    //     a.encode_vec_i64(i, log_base2k, log_k, &values, 32);
    //     a.normalize(log_base2k, i, &mut tmp_bytes_vmp);
    //     a.print(n, i);
    //     println!();
    // });

    // let mut mat_znx_dft: MatZnxDft<FFT64> = module.new_mat_znx_dft(mat_rows, mat_cols_in, mat_cols_out, mat_size);

    // (0..a.size()).for_each(|row_i| {
    //     let mut tmp: VecZnx = module.new_vec_znx(mat_cols_out, mat_size);
    //     (0..mat_cols_out).for_each(|j| {
    //         tmp.at_mut(j, row_i)[1 + j] = 1 as i64;
    //     });
    //     (0..mat_cols_in).for_each(|j| {
    //         module.vmp_prepare_row(&mut mat_znx_dft, row_i, j, &tmp, &mut tmp_bytes_vmp);
    //     })
    // });

    // let mut c_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft_from_bytes_borrow(mat_cols_out, mat_size, &mut tmp_bytes_dft);
    // module.vmp_apply_dft(&mut c_dft, &a, &mat_znx_dft, &mut tmp_bytes_vmp);

    // let mut res: VecZnx = module.new_vec_znx(mat_cols_out, a_size);
    // let mut c_big: VecZnxBig<FFT64> = c_dft.alias_as_vec_znx_big();
    // (0..mat_cols_out).for_each(|i| {
    //     module.vec_znx_idft_tmp_a(&mut c_big, i, &mut c_dft, i);
    //     module.vec_znx_big_normalize(log_base2k, &mut res, i, &c_big, i, &mut tmp_bytes_vmp);

    //     let mut values_res: Vec<i64> = vec![i64::default(); n];
    //     res.decode_vec_i64(i, log_base2k, log_k, &mut values_res);
    //     res.print(n, i);
    //     println!();
    //     println!("{:?}", values_res);
    //     println!();
    // });

    // module.free();
}
