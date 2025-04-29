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

    let mut carry: Vec<u8> = alloc_aligned(module.vec_znx_big_normalize_tmp_bytes());

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    // s <- Z_{-1, 0, 1}[X]/(X^{N}+1)
    let mut s: Scalar = Scalar::new(n);
    s.fill_ternary_prob(0.5, &mut source);

    // Buffer to store s in the DFT domain
    let mut s_ppol: ScalarZnxDft<FFT64> = module.new_svp_ppol();

    // s_ppol <- DFT(s)
    module.svp_prepare(&mut s_ppol, &s);

    // Allocates a VecZnx with two columns: ct=(0, 0)
    let mut ct: VecZnx = module.new_vec_znx(
        2,       // Number of columns
        ct_size, // Number of small poly per column
    );

    // Fill the second column with random values: ct = (0, a)
    module.fill_uniform(log_base2k, &mut ct, 1, ct_size, &mut source);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft<FFT64> = module.new_vec_znx_dft(
        1,         // Number of columns
        ct.size(), // Number of polynomials per column
    );

    // Applies DFT(ct[1]) * DFT(s)
    module.svp_apply_dft(
        &mut buf_dft, // DFT(ct[1] * s)
        &s_ppol,      // DFT(s)
        &ct,
        1, // Selects the second column of ct
    );

    // Alias scratch space (VecZnxDft<B> is always at least as big as VecZnxBig<B>)
    let mut buf_big: VecZnxBig<FFT64> = buf_dft.as_vec_znx_big();

    // BIG(ct[1] * s) <- IDFT(DFT(ct[1] * s)) (not normalized)
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft);

    // Creates a plaintext: VecZnx with 1 column
    let mut m: VecZnx = module.new_vec_znx(
        1,        // Number of columns
        msg_size, // Number of small polynomials
    );
    let mut want: Vec<i64> = vec![0; n];
    want.iter_mut()
        .for_each(|x| *x = source.next_u64n(16, 15) as i64);
    m.encode_vec_i64(0, log_base2k, log_scale, &want, 4);
    m.normalize(log_base2k, &mut carry);

    // m - BIG(ct[1] * s)
    module.vec_znx_big_sub_small_a_inplace(
        &mut buf_big,
        0, // Selects the first column of the receiver
        &m,
        0, // Selects the first column of the message
    );

    // Normalizes back to VecZnx
    // ct[0] <- m - BIG(c1 * s)
    module.vec_znx_big_normalize(
        log_base2k, &mut ct, 0, // Selects the first column of ct (ct[0])
        &buf_big, 0, // Selects the first column of buf_big
        &mut carry,
    );

    // Add noise to ct[0]
    // ct[0] <- ct[0] + e
    module.add_normal(
        log_base2k,
        &mut ct,
        0,                    // Selects the first column of ct (ct[0])
        log_base2k * ct_size, // Scaling of the noise: 2^{-log_base2k * limbs}
        &mut source,
        3.2,  // Standard deviation
        19.0, // Truncatation bound
    );

    // Final ciphertext: ct = (-a * s + m + e, a)

    // Decryption

    // DFT(ct[1] * s)
    module.svp_apply_dft(
        &mut buf_dft,
        &s_ppol,
        &ct,
        1, // Selects the second column of ct (ct[1])
    );

    // BIG(c1 * s) = IDFT(DFT(c1 * s))
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft);

    // BIG(c1 * s) + ct[0]
    module.vec_znx_big_add_small_inplace(&mut buf_big, 0, &ct, 0);

    // m + e <- BIG(ct[1] * s + ct[0])
    let mut res: VecZnx = module.new_vec_znx(1, ct_size);
    module.vec_znx_big_normalize(log_base2k, &mut res, 0, &buf_big, 0, &mut carry);

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
