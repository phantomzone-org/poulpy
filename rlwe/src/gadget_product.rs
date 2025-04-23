use crate::{ciphertext::Ciphertext, elem::ElemCommon, parameters::Parameters};
use base2k::{
    Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps,
};
use std::cmp::min;

pub fn gadget_product_core_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    res_log_q: usize,
    in_log_q: usize,
    gct_rows: usize,
    gct_log_q: usize,
) -> usize {
    let gct_cols: usize = (gct_log_q + log_base2k - 1) / log_base2k;
    let in_cols: usize = (in_log_q + log_base2k - 1) / log_base2k;
    let out_cols: usize = (res_log_q + log_base2k - 1) / log_base2k;
    module.vmp_apply_dft_to_dft_tmp_bytes(out_cols, in_cols, gct_rows, gct_cols)
}

impl Parameters {
    pub fn gadget_product_tmp_bytes(
        &self,
        res_log_q: usize,
        in_log_q: usize,
        gct_rows: usize,
        gct_log_q: usize,
    ) -> usize {
        gadget_product_core_tmp_bytes(
            self.module(),
            self.log_base2k(),
            res_log_q,
            in_log_q,
            gct_rows,
            gct_log_q,
        )
    }
}

pub fn gadget_product_core(
    module: &Module,
    res_dft_0: &mut VecZnxDft,
    res_dft_1: &mut VecZnxDft,
    a: &VecZnx,
    b: &Ciphertext<VmpPMat>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    assert!(b_cols <= b.cols());
    module.vec_znx_dft(res_dft_1, a);
    module.vmp_apply_dft_to_dft(res_dft_0, res_dft_1, b.at(0), tmp_bytes);
    module.vmp_apply_dft_to_dft_inplace(res_dft_1, b.at(1), tmp_bytes);
}

pub fn gadget_product_big_tmp_bytes(
    module: &Module,
    c_cols: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(min(c_cols, a_cols));
}

/// Evaluates the gadget product: c.at(i) = IDFT(<DFT(a.at(i)), b.at(i)>)
///
/// # Arguments
///
/// * `module`: backend support for operations mod (X^N + 1).
/// * `c`: a [Ciphertext<VecZnxBig>] with cols_c cols.
/// * `a`: a [Ciphertext<VecZnx>] with cols_a cols.
/// * `b`: a [Ciphertext<VmpPMat>] with at least min(cols_c, cols_a) rows.
pub fn gadget_product_big(
    module: &Module,
    c: &mut Ciphertext<VecZnxBig>,
    a: &Ciphertext<VecZnx>,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = min(c.cols(), a.cols());

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);

    // a1_dft = DFT(a[1])
    module.vec_znx_dft(&mut a1_dft, a.at(1));

    // c[i] = IDFT(DFT(a[1]) * b[i])
    (0..2).for_each(|i| {
        module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.at(i), tmp_bytes);
        module.vec_znx_idft_tmp_a(c.at_mut(i), &mut res_dft);
    })
}

/// Evaluates the gadget product: c.at(i) = NORMALIZE(IDFT(<DFT(a.at(i)), b.at(i)>)
///
/// # Arguments
///
/// * `module`: backend support for operations mod (X^N + 1).
/// * `c`: a [Ciphertext<VecZnx>] with cols_c cols.
/// * `a`: a [Ciphertext<VecZnx>] with cols_a cols.
/// * `b`: a [Ciphertext<VmpPMat>] with at least min(cols_c, cols_a) rows.
pub fn gadget_product(
    module: &Module,
    c: &mut Ciphertext<VecZnx>,
    a: &Ciphertext<VecZnx>,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = min(c.cols(), a.cols());

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // a1_dft = DFT(a[1])
    module.vec_znx_dft(&mut a1_dft, a.at(1));

    // c[i] = IDFT(DFT(a[1]) * b[i])
    (0..2).for_each(|i| {
        module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.at(i), tmp_bytes);
        module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
        module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(i), &mut res_big, tmp_bytes);
    })
}

#[cfg(test)]
mod test {
    use crate::{
        ciphertext::{Ciphertext, new_gadget_ciphertext},
        decryptor::decrypt_rlwe,
        elem::{Elem, ElemCommon, ElemVecZnx},
        encryptor::encrypt_grlwe_sk,
        gadget_product::gadget_product_core,
        keys::SecretKey,
        parameters::{Parameters, ParametersLiteral},
        plaintext::Plaintext,
    };
    use base2k::{
        BACKEND, Infos, Sampling, SvpPPolOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft,
        VecZnxDftOps, VecZnxOps, VmpPMat, alloc_aligned_u8,
    };
    use sampling::source::{Source, new_seed};

    #[test]
    fn test_gadget_product_core() {
        let log_base2k: usize = 10;
        let q_cols: usize = 7;
        let p_cols: usize = 1;

        // Basic parameters with enough limbs to test edge cases
        let params_lit: ParametersLiteral = ParametersLiteral {
            backend: BACKEND::FFT64,
            log_n: 12,
            log_q: q_cols * log_base2k,
            log_p: p_cols * log_base2k,
            log_base2k: log_base2k,
            log_scale: 20,
            xe: 3.2,
            xs: 1 << 11,
        };

        let params: Parameters = Parameters::new(&params_lit);

        // scratch space
        let mut tmp_bytes: Vec<u8> = alloc_aligned_u8(
            params.decrypt_rlwe_tmp_byte(params.log_qp())
                | params.gadget_product_tmp_bytes(
                    params.log_qp(),
                    params.log_qp(),
                    params.cols_qp(),
                    params.log_qp(),
                )
                | params.encrypt_grlwe_sk_tmp_bytes(params.cols_qp(), params.log_qp()),
        );

        // Samplers for public and private randomness
        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xs: Source = Source::new(new_seed());

        // Two secret keys
        let mut sk0: SecretKey = SecretKey::new(params.module());
        sk0.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk0_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
        params.module().svp_prepare(&mut sk0_svp_ppol, &sk0.0);

        let mut sk1: SecretKey = SecretKey::new(params.module());
        sk1.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk1_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
        params.module().svp_prepare(&mut sk1_svp_ppol, &sk1.0);

        // The gadget ciphertext
        let mut gadget_ct: Ciphertext<VmpPMat> = new_gadget_ciphertext(
            params.module(),
            log_base2k,
            params.cols_qp(),
            params.log_qp(),
        );

        // gct = [-b*sk1 + g(sk0) + e, b]
        encrypt_grlwe_sk(
            params.module(),
            &mut gadget_ct,
            &sk0.0,
            &sk1_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        // Intermediate buffers

        // Input polynopmial, uniformly distributed
        let mut a: VecZnx = params.module().new_vec_znx(params.cols_q());
        params
            .module()
            .fill_uniform(log_base2k, &mut a, params.cols_q(), &mut source_xa);

        // res = g^-1(a) * gct
        let mut elem_res: Elem<VecZnx> =
            Elem::<VecZnx>::new(params.module(), log_base2k, params.log_qp(), 2);

        // Ideal output = a * s
        let mut a_dft: VecZnxDft = params.module().new_vec_znx_dft(a.cols());
        let mut a_big: VecZnxBig = a_dft.as_vec_znx_big();
        let mut a_times_s: VecZnx = params.module().new_vec_znx(a.cols());

        // a * sk0
        params.module().svp_apply_dft(&mut a_dft, &sk0_svp_ppol, &a);
        params.module().vec_znx_idft_tmp_a(&mut a_big, &mut a_dft);
        params.module().vec_znx_big_normalize(
            params.log_base2k(),
            &mut a_times_s,
            &a_big,
            &mut tmp_bytes,
        );

        // Plaintext for decrypted output of gadget product
        let mut pt: Plaintext =
            Plaintext::new(params.module(), params.log_base2k(), params.log_qp());

        // Iterates over all possible cols values for input/output polynomials and gadget ciphertext.

        (1..a.cols() + 1).for_each(|a_cols| {
            let mut a_trunc: VecZnx = params.module().new_vec_znx(a_cols);
            a_trunc.copy_from(&a);

            (1..gadget_ct.cols() + 1).for_each(|b_cols| {
                let mut res_dft_0: VecZnxDft = params.module().new_vec_znx_dft(b_cols);
                let mut res_dft_1: VecZnxDft = params.module().new_vec_znx_dft(b_cols);
                let mut res_big_0: VecZnxBig = res_dft_0.as_vec_znx_big();
                let mut res_big_1: VecZnxBig = res_dft_1.as_vec_znx_big();

                pt.elem_mut().zero();
                elem_res.zero();

                //let b_cols: usize = min(a_cols+1, gadget_ct.cols());

                println!("a_cols: {} b_cols: {}", a_cols, b_cols);

                // res_dft_0 = DFT(gct_[0] * ct[1] = a * (-bs' + s + e) = -cs' + as + e')
                // res_dft_1 = DFT(gct_[1] * ct[1] = a * b = c)
                gadget_product_core(
                    params.module(),
                    &mut res_dft_0,
                    &mut res_dft_1,
                    &a_trunc,
                    &gadget_ct,
                    b_cols,
                    &mut tmp_bytes,
                );

                // res_big_0 = IDFT(res_dft_0)
                params
                    .module()
                    .vec_znx_idft_tmp_a(&mut res_big_0, &mut res_dft_0);
                // res_big_1 = IDFT(res_dft_1);
                params
                    .module()
                    .vec_znx_idft_tmp_a(&mut res_big_1, &mut res_dft_1);

                // res_big_0 = normalize(res_big_0)
                params.module().vec_znx_big_normalize(
                    log_base2k,
                    elem_res.at_mut(0),
                    &res_big_0,
                    &mut tmp_bytes,
                );

                // res_big_1 = normalize(res_big_1)
                params.module().vec_znx_big_normalize(
                    log_base2k,
                    elem_res.at_mut(1),
                    &res_big_1,
                    &mut tmp_bytes,
                );

                // <(-c*sk1 + a*sk0 + e, a), (1, sk1)> = a*sk0 + e
                decrypt_rlwe(
                    params.module(),
                    pt.elem_mut(),
                    &elem_res,
                    &sk1_svp_ppol,
                    &mut tmp_bytes,
                );

                // a * sk0 + e - a*sk0 = e
                params
                    .module()
                    .vec_znx_sub_ab_inplace(pt.at_mut(0), &mut a_times_s);
                pt.at_mut(0).normalize(log_base2k, &mut tmp_bytes);

                //pt.at(0).print(pt.elem().cols(), 16);

                let noise_have: f64 = pt.at(0).std(log_base2k).log2();

                let var_a_err: f64;

                if a_cols < a.cols() {
                    var_a_err = 1f64 / 12f64;
                } else {
                    var_a_err = 0f64;
                }

                let a_logq: usize = a_cols * log_base2k;
                let b_logq: usize = b_cols * log_base2k;
                let var_msg: f64 = (params.xs() as f64) / params.n() as f64;

                println!("{} {} {} {}", var_msg, var_a_err, a_logq, b_logq);

                let noise_pred: f64 =
                    params.noise_grlwe_product(var_msg, var_a_err, a_logq, b_logq);

                println!("noise_pred: {}", noise_pred);
                println!("noise_have: {}", noise_have);

                //assert!(noise_have <= noise_pred + 1.0);
            });
        });
    }
}

impl Parameters {
    pub fn noise_grlwe_product(
        &self,
        var_msg: f64,
        var_a_err: f64,
        a_logq: usize,
        b_logq: usize,
    ) -> f64 {
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

        noise_grlwe_product(
            n,
            self.log_base2k(),
            var_xs,
            var_msg,
            var_a_err,
            var_gct_err_lhs,
            var_gct_err_rhs,
            a_logq,
            b_logq,
        )
    }
}

pub fn noise_grlwe_product(
    n: f64,
    log_base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
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
    let mut noise: f64 =
        (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * a_scale * a_scale * n;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
