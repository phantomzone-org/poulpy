use crate::{ciphertext::Ciphertext, elem::ElemCommon, parameters::Parameters};
use base2k::{Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, MatZnxDft, MatZnxDftOps, assert_alignement};
use std::cmp::min;

impl Parameters {
    pub fn rgsw_product_tmp_bytes(&self, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
        rgsw_product_tmp_bytes(
            self.module(),
            self.log_base2k(),
            res_logq,
            in_logq,
            gct_logq,
        )
    }
}
pub fn rgsw_product_tmp_bytes(module: &Module, log_base2k: usize, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
    let gct_cols: usize = (gct_logq + log_base2k - 1) / log_base2k;
    let in_cols: usize = (in_logq + log_base2k - 1) / log_base2k;
    let res_cols: usize = (res_logq + log_base2k - 1) / log_base2k;
    return module.vmp_apply_dft_to_dft_tmp_bytes(res_cols, in_cols, in_cols, gct_cols)
        + module.bytes_of_vec_znx_dft(1, std::cmp::min(res_cols, in_cols))
        + 2 * module.bytes_of_vec_znx_dft(1, gct_cols);
}

pub fn rgsw_product(
    module: &Module,
    c: &mut Ciphertext<VecZnx>,
    a: &Ciphertext<VecZnx>,
    b: &Ciphertext<MatZnxDft>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    #[cfg(debug_assertions)]
    {
        assert!(b_cols <= b.cols());
        assert_eq!(c.size(), 2);
        assert_eq!(a.size(), 2);
        assert_eq!(b.size(), 4);
        assert!(tmp_bytes.len() >= rgsw_product_tmp_bytes(module, c.cols(), a.cols(), min(b.rows(), a.cols()), b_cols));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_ai_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, a.cols()));
    let (tmp_bytes_c0_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, b_cols));
    let (tmp_bytes_c1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, b_cols));

    let mut ai_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, a.cols(), tmp_bytes_ai_dft);
    let mut c0_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, b_cols, tmp_bytes_c0_dft);
    let mut c1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, b_cols, tmp_bytes_c1_dft);

    let mut c0_big: VecZnxBig = c0_dft.as_vec_znx_big();
    let mut c1_big: VecZnxBig = c1_dft.as_vec_znx_big();

    module.vec_znx_dft(&mut ai_dft, a.at(0));
    module.vmp_apply_dft_to_dft(&mut c0_dft, &ai_dft, b.at(0), tmp_bytes);
    module.vmp_apply_dft_to_dft(&mut c1_dft, &ai_dft, b.at(1), tmp_bytes);

    module.vec_znx_dft(&mut ai_dft, a.at(1));
    module.vmp_apply_dft_to_dft_add(&mut c0_dft, &ai_dft, b.at(2), tmp_bytes);
    module.vmp_apply_dft_to_dft_add(&mut c1_dft, &ai_dft, b.at(3), tmp_bytes);

    module.vec_znx_idft_tmp_a(&mut c0_big, &mut c0_dft);
    module.vec_znx_idft_tmp_a(&mut c1_big, &mut c1_dft);

    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(0), &mut c0_big, tmp_bytes);
    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(1), &mut c1_big, tmp_bytes);
}

pub fn rgsw_product_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    b: &Ciphertext<MatZnxDft>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    #[cfg(debug_assertions)]
    {
        assert!(b_cols <= b.cols());
        assert_eq!(a.size(), 2);
        assert_eq!(b.size(), 4);
        assert!(tmp_bytes.len() >= rgsw_product_tmp_bytes(module, a.cols(), a.cols(), min(b.rows(), a.cols()), b_cols));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_ai_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, a.cols()));
    let (tmp_bytes_c0_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, b_cols));
    let (tmp_bytes_c1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(1, b_cols));

    let mut ai_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, a.cols(), tmp_bytes_ai_dft);
    let mut c0_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, b_cols, tmp_bytes_c0_dft);
    let mut c1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(1, b_cols, tmp_bytes_c1_dft);

    let mut c0_big: VecZnxBig = c0_dft.as_vec_znx_big();
    let mut c1_big: VecZnxBig = c1_dft.as_vec_znx_big();

    module.vec_znx_dft(&mut ai_dft, a.at(0));
    module.vmp_apply_dft_to_dft(&mut c0_dft, &ai_dft, b.at(0), tmp_bytes);
    module.vmp_apply_dft_to_dft(&mut c1_dft, &ai_dft, b.at(1), tmp_bytes);

    module.vec_znx_dft(&mut ai_dft, a.at(1));
    module.vmp_apply_dft_to_dft_add(&mut c0_dft, &ai_dft, b.at(2), tmp_bytes);
    module.vmp_apply_dft_to_dft_add(&mut c1_dft, &ai_dft, b.at(3), tmp_bytes);

    module.vec_znx_idft_tmp_a(&mut c0_big, &mut c0_dft);
    module.vec_znx_idft_tmp_a(&mut c1_big, &mut c1_dft);

    module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(0), &mut c0_big, tmp_bytes);
    module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(1), &mut c1_big, tmp_bytes);
}

#[cfg(test)]
mod test {
    use crate::{
        ciphertext::{Ciphertext, new_rgsw_ciphertext},
        decryptor::decrypt_rlwe,
        elem::ElemCommon,
        encryptor::{encrypt_rgsw_sk, encrypt_rlwe_sk},
        keys::SecretKey,
        parameters::{DEFAULT_SIGMA, Parameters, ParametersLiteral},
        plaintext::Plaintext,
        rgsw_product::rgsw_product_inplace,
    };
    use base2k::{BACKEND, Encoding, Module, Scalar, ScalarZnxDft, ScalarZnxDftOps, VecZnx, VecZnxOps, MatZnxDft, alloc_aligned};
    use sampling::source::{Source, new_seed};

    #[test]
    fn test_rgsw_product() {
        let log_base2k: usize = 10;
        let log_q: usize = 50;
        let log_p: usize = 15;

        // Basic parameters with enough limbs to test edge cases
        let params_lit: ParametersLiteral = ParametersLiteral {
            backend: BACKEND::FFT64,
            log_n: 12,
            log_q: log_q,
            log_p: log_p,
            log_base2k: log_base2k,
            log_scale: 20,
            xe: 3.2,
            xs: 1 << 11,
        };

        let params: Parameters = Parameters::new(&params_lit);

        let module: &Module = params.module();
        let log_q: usize = params.log_q();
        let log_qp: usize = params.log_qp();
        let gct_rows: usize = params.cols_q();
        let gct_cols: usize = params.cols_qp();

        // scratch space
        let mut tmp_bytes: Vec<u8> = alloc_aligned(
            params.decrypt_rlwe_tmp_byte(log_q)
                | params.encrypt_rlwe_sk_tmp_bytes(log_q)
                | params.rgsw_product_tmp_bytes(log_q, log_q, log_qp)
                | params.encrypt_rgsw_sk_tmp_bytes(gct_rows, log_qp),
        );

        // Samplers for public and private randomness
        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xs: Source = Source::new(new_seed());

        let mut sk: SecretKey = SecretKey::new(module);
        sk.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk_svp_ppol: ScalarZnxDft = module.new_svp_ppol();
        module.svp_prepare(&mut sk_svp_ppol, &sk.0);

        let mut ct_rgsw: Ciphertext<MatZnxDft> = new_rgsw_ciphertext(module, log_base2k, gct_rows, log_qp);

        let k: i64 = 3;

        // X^k
        let m: Scalar = module.new_scalar();
        let data: &mut [i64] = m.raw_mut();
        data[k as usize] = 1;

        encrypt_rgsw_sk(
            module,
            &mut ct_rgsw,
            &m,
            &sk_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            DEFAULT_SIGMA,
            &mut tmp_bytes,
        );

        let log_k: usize = 2 * log_base2k;

        let mut ct: Ciphertext<VecZnx> = params.new_ciphertext(log_q);
        let mut pt: Plaintext = params.new_plaintext(log_q);
        let mut pt_rotate: Plaintext = params.new_plaintext(log_q);

        pt.at_mut(0).encode_vec_i64(0, log_base2k, log_k, &data, 32);

        module.vec_znx_rotate(k, pt_rotate.at_mut(0), pt.at_mut(0));

        encrypt_rlwe_sk(
            module,
            &mut ct.elem_mut(),
            Some(pt.at(0)),
            &sk_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        rgsw_product_inplace(module, &mut ct, &ct_rgsw, gct_cols, &mut tmp_bytes);

        decrypt_rlwe(
            module,
            pt.elem_mut(),
            ct.elem(),
            &sk_svp_ppol,
            &mut tmp_bytes,
        );

        module.vec_znx_sub_ba_inplace(pt.at_mut(0), pt_rotate.at(0));

        // pt.at(0).print(pt.cols(), 16);

        let noise_have: f64 = pt.at(0).std(0, log_base2k).log2();

        let var_msg: f64 = 1f64 / params.n() as f64; // X^{k}
        let var_a0_err: f64 = params.xe() * params.xe();
        let var_a1_err: f64 = 1f64 / 12f64;

        let noise_pred: f64 = params.noise_rgsw_product(var_msg, var_a0_err, var_a1_err, ct.log_q(), ct_rgsw.log_q());

        println!("noise_pred: {}", noise_pred);
        println!("noise_have: {}", noise_have);

        assert!(noise_have <= noise_pred + 1.0);
    }
}

impl Parameters {
    pub fn noise_rgsw_product(&self, var_msg: f64, var_a0_err: f64, var_a1_err: f64, a_logq: usize, b_logq: usize) -> f64 {
        let n: f64 = self.n() as f64;
        let var_xs: f64 = self.xs() as f64;

        let var_gct_err_lhs: f64;
        let var_gct_err_rhs: f64;
        if b_logq < self.log_qp() {
            let var_round: f64 = 1f64 / 12f64;
            var_gct_err_lhs = var_round;
            var_gct_err_rhs = var_round;
        } else {
            var_gct_err_lhs = self.xe() * self.xe();
            var_gct_err_rhs = 0f64;
        }

        noise_rgsw_product(
            n,
            self.log_base2k(),
            var_xs,
            var_msg,
            var_a0_err,
            var_a1_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            a_logq,
            b_logq,
        )
    }
}

pub fn noise_rgsw_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a0_err: f64,
    var_a1_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let a_logq: usize = min(a_logq, b_logq);
    let a_cols: usize = (a_logq + log_base2k - 1) / log_base2k;

    let b_scale = 2.0f64.powi(b_logq as i32);
    let a_scale: f64 = 2.0f64.powi((b_logq - a_logq) as i32);

    let base: f64 = (1 << (log_base2k)) as f64;
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = 2.0 * (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a0_err * a_scale * a_scale * n;
    noise += var_msg * var_a1_err * a_scale * a_scale * n * var_xs;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
