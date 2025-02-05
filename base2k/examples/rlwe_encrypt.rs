use base2k::{
    Encoding, Infos, Module, Sampling, Scalar, SvpPPol, SvpPPolOps, VecZnx, VecZnxBig, VecZnxDft,
    VecZnxOps, FFT64,
};
use itertools::izip;
use sampling::source::Source;

fn main() {
    let n: usize = 16;
    let log_base2k: usize = 18;
    let limbs: usize = 3;
    let msg_limbs: usize = 2;
    let log_scale: usize = msg_limbs * log_base2k - 5;
    let module: Module = Module::new::<FFT64>(n);

    let mut carry: Vec<u8> = vec![0; module.vec_znx_big_normalize_tmp_bytes()];

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let mut res: VecZnx = module.new_vec_znx(limbs);

    // s <- Z_{-1, 0, 1}[X]/(X^{N}+1)
    let mut s: Scalar = Scalar::new(n);
    s.fill_ternary_prob(0.5, &mut source);

    // Buffer to store s in the DFT domain
    let mut s_ppol: SvpPPol = module.svp_new_ppol();

    // s_ppol <- DFT(s)
    module.svp_prepare(&mut s_ppol, &s);

    // a <- Z_{2^prec}[X]/(X^{N}+1)
    let mut a: VecZnx = module.new_vec_znx(limbs);
    a.fill_uniform(log_base2k, limbs, &mut source);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft = module.new_vec_znx_dft(a.limbs());

    // Applies buf_dft <- s * a
    module.svp_apply_dft(&mut buf_dft, &s_ppol, &a);

    // Alias scratch space
    let mut buf_big: VecZnxBig = buf_dft.as_vec_znx_big();

    // buf_big <- IDFT(buf_dft) (not normalized)
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, a.limbs());

    let mut m: VecZnx = module.new_vec_znx(msg_limbs);

    let mut want: Vec<i64> = vec![0; n];
    want.iter_mut()
        .for_each(|x| *x = source.next_u64n(16, 15) as i64);

    // m
    m.encode_vec_i64(log_base2k, log_scale, &want, 4);
    m.normalize(log_base2k, &mut carry);

    // buf_big <- m - buf_big
    module.vec_znx_big_sub_small_a_inplace(&mut buf_big, &m);

    // b <- normalize(buf_big) + e
    let mut b: VecZnx = module.new_vec_znx(limbs);
    module.vec_znx_big_normalize(log_base2k, &mut b, &buf_big, &mut carry);
    b.add_normal(log_base2k, log_base2k * limbs, &mut source, 3.2, 19.0);

    //Decrypt

    // buf_big <- a * s
    module.svp_apply_dft(&mut buf_dft, &s_ppol, &a);
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, b.limbs());

    // buf_big <- a * s + b
    module.vec_znx_big_add_small_inplace(&mut buf_big, &b);

    // res <- normalize(buf_big)
    module.vec_znx_big_normalize(log_base2k, &mut res, &buf_big, &mut carry);

    // have = m * 2^{log_scale} + e
    let mut have: Vec<i64> = vec![i64::default(); n];
    res.decode_vec_i64(log_base2k, res.limbs() * log_base2k, &mut have);

    let scale: f64 = (1 << (res.limbs() * log_base2k - log_scale)) as f64;
    izip!(want.iter(), have.iter())
        .enumerate()
        .for_each(|(i, (a, b))| {
            println!("{}: {} {}", i, a, (*b as f64) / scale);
        })
}
