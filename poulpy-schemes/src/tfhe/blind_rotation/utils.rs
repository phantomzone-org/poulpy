use poulpy_backend::hal::{
    api::{SvpPrepare, ZnxInfos, ZnxViewMut},
    layouts::{Backend, DataMut, Module, ScalarZnx, SvpPPol},
};

pub(crate) fn set_xai_plus_y<A, C, B: Backend>(
    module: &Module<B>,
    ai: usize,
    y: i64,
    res: &mut SvpPPol<A, B>,
    buf: &mut ScalarZnx<C>,
) where
    A: DataMut,
    C: DataMut,
    Module<B>: SvpPrepare<B>,
{
    let n: usize = res.n();

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);
        if ai < n {
            raw[ai] = 1;
        } else {
            raw[(ai - n) & (n - 1)] = -1;
        }
        raw[0] += y;
    }

    module.svp_prepare(res, 0, buf, 0);

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);

        if ai < n {
            raw[ai] = 0;
        } else {
            raw[(ai - n) & (n - 1)] = 0;
        }
        raw[0] = 0;
    }
}
