use rug::{
    Float,
    float::Round,
    ops::{AddAssignRound, DivAssignRound, SubAssignRound},
};

use crate::layouts::{Backend, DataRef, VecZnx, VecZnxBig, VecZnxBigToRef, ZnxInfos};

pub struct Stats {
    max: f64,
    std: f64,
}

impl Stats {
    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn std(&self) -> f64 {
        self.std
    }
}

impl<D: DataRef> VecZnx<D> {
    pub fn stats(&self, base2k: usize, col: usize) -> Stats {
        let prec: u32 = (self.size() * base2k) as u32;
        let mut data: Vec<Float> = (0..self.n()).map(|_| Float::with_val(prec, 0)).collect();
        self.decode_vec_float(base2k, col, &mut data);
        // std = sqrt(sum((xi - avg)^2) / n)
        let mut avg: Float = Float::with_val(prec, 0);
        let mut max: Float = Float::with_val(prec, 0);

        data.iter().for_each(|x| {
            avg.add_assign_round(x, Round::Nearest);
            max.max_mut(&Float::with_val(53, x.abs_ref()));
        });
        avg.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        data.iter_mut().for_each(|x| {
            x.sub_assign_round(&avg, Round::Nearest);
        });
        let mut std: Float = Float::with_val(prec, 0);
        data.iter().for_each(|x| std += x * x);
        std.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        std = std.sqrt();
        Stats {
            std: std.to_f64(),
            max: max.to_f64(),
        }
    }
}

impl<D: DataRef, B: Backend + Backend<ScalarBig = i64>> VecZnxBig<D, B> {
    pub fn stats(&self, base2k: usize, col: usize) -> Stats {
        let self_ref: VecZnxBig<&[u8], B> = self.to_ref();
        let znx: VecZnx<&[u8]> = VecZnx {
            data: self_ref.data,
            n: self_ref.n,
            cols: self_ref.cols,
            size: self_ref.size,
            max_size: self_ref.max_size,
        };
        znx.stats(base2k, col)
    }
}
