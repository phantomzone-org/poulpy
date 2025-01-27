use crate::poly::Poly;
use crate::ring::Ring;

impl Ring<u64> {
    pub fn switch_degree<const NTT: bool>(
        &self,
        a: &Poly<u64>,
        buf: &mut Poly<u64>,
        b: &mut Poly<u64>,
    ) {
        let (n_in, n_out) = (a.n(), b.n());

        if n_in > n_out {
            let (gap_in, gap_out) = (1, n_in / n_out);
            if NTT {
                self.intt::<false>(&a, buf);
                b.0.iter_mut()
                    .step_by(gap_in)
                    .zip(buf.0.iter().step_by(gap_out))
                    .for_each(|(x_out, x_in)| *x_out = *x_in);
                self.ntt_inplace::<false>(b);
            } else {
                b.0.iter_mut()
                    .step_by(gap_in)
                    .zip(a.0.iter().step_by(gap_out))
                    .for_each(|(x_out, x_in)| *x_out = *x_in);
            }
        } else {
            let gap: usize = n_out / n_in;

            if NTT {
                a.0.iter()
                    .enumerate()
                    .for_each(|(i, &c)| (0..gap).for_each(|j| b.0[i * gap + j] = c));
            } else {
                b.0.iter_mut()
                    .step_by(gap)
                    .zip(a.0.iter())
                    .for_each(|(x_out, x_in)| *x_out = *x_in);
            }
        }
    }
}
