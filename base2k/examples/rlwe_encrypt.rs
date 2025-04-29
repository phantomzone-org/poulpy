use base2k::{
    Encoding, FFT64, Module, Sampling, Scalar, ScalarZnxDft, ScalarZnxDftOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft,
    VecZnxDftOps, VecZnxOps, ZnxInfos, alloc_aligned,
};
use itertools::izip;
use sampling::source::Source;

fn main() {
    let n: usize = 16;
    let log_base2k: usize = 18;
    let ct_size: usize = 3;
    let msg_size: usize = 2;
    let log_scale: usize = msg_size * log_base2k - 5;
    let module: Module<FFT64> = Module::<FFT64>::new(n);

    let mut carry: Vec<u8> = alloc_aligned(module.vec_znx_big_normalize_tmp_bytes(2));

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    // s <- Z_{-1, 0, 1}[X]/(X^{N}+1)
    let mut s: Scalar = Scalar::new(n);
    s.fill_ternary_prob(0.5, &mut source);

    // Buffer to store s in the DFT domain
    let mut s_ppol: ScalarZnxDft<FFT64> = module.new_svp_ppol();

    // s_ppol <- DFT(s)
    module.svp_prepare(&mut s_ppol, &s);

    // ct = (c0, c1)
    let mut ct: VecZnx = module.new_vec_znx(2, ct_size);

    // Fill c1 with random values
    module.fill_uniform(log_base2k, &mut ct, 1, ct_size, &mut source);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(1, ct.size());

    // Applies buf_dft <- s * c1
    module.svp_apply_dft(
        &mut buf_dft, // DFT(c1 * s)
        &s_ppol,
        &ct,
        1, // c1
    );

    // Alias scratch space (VecZnxDftis always at least as big as VecZnxBig)
    let mut buf_big: VecZnxBig<FFT64> = buf_dft.as_vec_znx_big();

    // BIG(c1 * s) <- IDFT(DFT(c1 * s)) (not normalized)
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft);

    // m <- (0)
    let mut m: VecZnx = module.new_vec_znx(1, msg_size);
    let mut want: Vec<i64> = vec![0; n];
    want.iter_mut()
        .for_each(|x| *x = source.next_u64n(16, 15) as i64);
    m.encode_vec_i64(0, log_base2k, log_scale, &want, 4);
    m.normalize(log_base2k, &mut carry);

    // m - BIG(c1 * s)
    module.vec_znx_big_sub_small_ab_inplace(&mut buf_big, &m);

    // c0 <- m - BIG(c1 * s)
    module.vec_znx_big_normalize(log_base2k, &mut ct, &buf_big, &mut carry);

    ct.print(ct.sl());

    // (c0 + e, c1)
    module.add_normal(
        log_base2k,
        &mut ct,
        0, // c0
        log_base2k * ct_size,
        &mut source,
        3.2,
        19.0,
    );

    // Decrypt

    // DFT(c1 * s)
    module.svp_apply_dft(&mut buf_dft, &s_ppol, &ct, 1);
    // BIG(c1 * s) = IDFT(DFT(c1 * s))
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft);

    // BIG(c1 * s) + c0
    module.vec_znx_big_add_small_inplace(&mut buf_big, &ct);

    // m + e <- BIG(c1 * s + c0)
    let mut res: VecZnx = module.new_vec_znx(1, ct_size);
    module.vec_znx_big_normalize(log_base2k, &mut res, &buf_big, &mut carry);

    // have = m * 2^{log_scale} + e
    let mut have: Vec<i64> = vec![i64::default(); n];
    res.decode_vec_i64(0, log_base2k, res.size() * log_base2k, &mut have);

    let scale: f64 = (1 << (res.size() * log_base2k - log_scale)) as f64;
    izip!(want.iter(), have.iter())
        .enumerate()
        .for_each(|(i, (a, b))| {
            println!("{}: {} {}", i, a, (*b as f64) / scale);
        })
}
