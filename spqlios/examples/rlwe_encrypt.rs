use itertools::izip;
use sampling::source::Source;
use spqlios::module::{Module, FFT64};
use spqlios::scalar::Scalar;
use spqlios::vector::Vector;

fn main() {
    let n: usize = 16;
    let log_base2k: usize = 40;
    let prec: usize = 54;
    let log_scale: usize = 18;
    let module: Module = Module::new::<FFT64>(n);

    let mut carry: Vec<u8> = vec![0; module.vec_znx_big_normalize_tmp_bytes()];

    let seed: [u8; 32] = [0; 32];
    let mut source: Source = Source::new(seed);

    let mut res: Vector = Vector::new(n, log_base2k, prec);

    // s <- Z_{-1, 0, 1}[X]/(X^{N}+1)
    let mut s: Scalar = Scalar::new(n);
    s.fill_ternary_prob(0.5, &mut source);

    // Buffer to store s in the DFT domain
    let mut s_ppol: spqlios::module::SVPPOL = module.svp_new_ppol();

    // s_ppol <- DFT(s)
    module.svp_prepare(&mut s_ppol, &s);

    // a <- Z_{2^prec}[X]/(X^{N}+1)
    let mut a: Vector = Vector::new(n, log_base2k, prec);
    a.fill_uniform(&mut source);

    // Scratch space for DFT values
    let mut buf_dft: spqlios::module::VECZNXDFT = module.new_vec_znx_dft(a.limbs());

    // Applies buf_dft <- s * a
    module.svp_apply_dft(&mut buf_dft, &s_ppol, &a);

    // Alias scratch space
    let mut buf_big: spqlios::module::VECZNXBIG = buf_dft.as_vec_znx_big();

    // buf_big <- IDFT(buf_dft) (not normalized)
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, a.limbs());

    let mut m: Vector = Vector::new(n, log_base2k, prec - log_scale);
    let mut want: Vec<i64> = vec![0; n];
    want.iter_mut()
        .for_each(|x| *x = source.next_u64n(16, 15) as i64);

    // m
    m.set_i64(&want, 4);
    m.normalize(&mut carry);

    // buf_big <- m - buf_big
    module.vec_znx_big_sub_small_a_inplace(&mut buf_big, &m);

    // b <- normalize(buf_big) + e
    let mut b: Vector = Vector::new(n, log_base2k, prec);
    module.vec_znx_big_normalize(&mut b, &buf_big, &mut carry);
    b.add_normal(&mut source, 3.2, 19.0);

    //Decrypt

    // buf_big <- a * s
    module.svp_apply_dft(&mut buf_dft, &s_ppol, &a);
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, b.limbs());

    // buf_big <- a * s + b
    module.vec_znx_big_add_small_inplace(&mut buf_big, &b);

    // res <- normalize(buf_big)
    module.vec_znx_big_normalize(&mut res, &buf_big, &mut carry);

    // have = m * 2^{log_scale} + e
    let mut have: Vec<i64> = vec![i64::default(); n];
    res.get_i64(&mut have);

    let scale: f64 = (1 << log_scale) as f64;
    izip!(want.iter(), have.iter())
        .enumerate()
        .for_each(|(i, (a, b))| {
            println!("{}: {} {}", i, a, (*b as f64) / scale);
        })
}
