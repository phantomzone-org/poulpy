#![feature(bigint_helper_methods)]
#![feature(test)]

pub mod modulus;
pub mod dft;
pub mod ring;
pub mod poly;

pub const CHUNK: usize= 8;

pub mod macros{
    
    #[macro_export]
    macro_rules! apply_v {

        ($self:expr, $f:expr, $a:expr, $CHUNK:expr) => {

            match CHUNK{
                8 => {
                    $a.chunks_exact_mut(8).for_each(|a| {
                        $f(&$self, &mut a[0]);
                        $f(&$self, &mut a[1]);
                        $f(&$self, &mut a[2]);
                        $f(&$self, &mut a[3]);
                        $f(&$self, &mut a[4]);
                        $f(&$self, &mut a[5]);
                        $f(&$self, &mut a[6]);
                        $f(&$self, &mut a[7]);
                    });

                    let n: usize = $a.len();
                    let m = n - (n&(CHUNK-1));
                    $a[m..].iter_mut().for_each(|a| {
                        $f(&$self, a);
                    });
                },
                _=>{
                    $a.iter_mut().for_each(|a| {
                        $f(&$self, a);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vv {

        ($self:expr, $f:expr, $a:expr, $b:expr, $CHUNK:expr) => {

            let n: usize = $a.len();
            debug_assert!($b.len() == n, "invalid argument b: b.len() = {} != a.len() = {}", $b.len(), n);
            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($a.chunks_exact(8), $b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        $f(&$self, &a[0], &mut b[0]);
                        $f(&$self, &a[1], &mut b[1]);
                        $f(&$self, &a[2], &mut b[2]);
                        $f(&$self, &a[3], &mut b[3]);
                        $f(&$self, &a[4], &mut b[4]);
                        $f(&$self, &a[5], &mut b[5]);
                        $f(&$self, &a[6], &mut b[6]);
                        $f(&$self, &a[7], &mut b[7]);
                    });

                    let m = n - (n&(CHUNK-1));
                    izip!($a[m..].iter(), $b[m..].iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, b);
                    });
                },
                _=>{
                    izip!($a.iter(), $b.iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, b);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vvv {

        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $CHUNK:expr) => {

            let n: usize = $a.len();
            debug_assert!($b.len() == n, "invalid argument b: b.len() = {} != a.len() = {}", $b.len(), n);
            debug_assert!($c.len() == n, "invalid argument c: b.len() = {} != a.len() = {}", $c.len(), n);
            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($a.chunks_exact(8), $b.chunks_exact(8), $c.chunks_exact_mut(8)).for_each(|(a, b, c)| {
                        $f(&$self, &a[0], &b[0], &mut c[0]);
                        $f(&$self, &a[1], &b[1], &mut c[1]);
                        $f(&$self, &a[2], &b[2], &mut c[2]);
                        $f(&$self, &a[3], &b[3], &mut c[3]);
                        $f(&$self, &a[4], &b[4], &mut c[4]);
                        $f(&$self, &a[5], &b[5], &mut c[5]);
                        $f(&$self, &a[6], &b[6], &mut c[6]);
                        $f(&$self, &a[7], &b[7], &mut c[7]);
                    });

                    let m = n - (n&7);
                    izip!($a[m..].iter(), $b[m..].iter(), $c[m..].iter_mut()).for_each(|(a, b, c)| {
                        $f(&$self, a, b, c);
                    });
                },
                _=>{
                    izip!($a.iter(), $b.iter(), $c.iter_mut()).for_each(|(a, b, c)| {
                        $f(&$self, a, b, c);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_sv {

        ($self:expr, $f:expr, $a:expr, $b:expr, $CHUNK:expr) => {

            let n: usize = $b.len();

            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($b.chunks_exact_mut(8)).for_each(|b| {
                        $f(&$self, $a, &mut b[0]);
                        $f(&$self, $a, &mut b[1]);
                        $f(&$self, $a, &mut b[2]);
                        $f(&$self, $a, &mut b[3]);
                        $f(&$self, $a, &mut b[4]);
                        $f(&$self, $a, &mut b[5]);
                        $f(&$self, $a, &mut b[6]);
                        $f(&$self, $a, &mut b[7]);
                    });

                    let m = n - (n&7);
                    izip!($b[m..].iter_mut()).for_each(|b| {
                        $f(&$self, $a, b);
                    });
                },
                _=>{
                    izip!($b.iter_mut()).for_each(|b| {
                        $f(&$self, $a, b);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_svv {

        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $CHUNK:expr) => {

            let n: usize = $b.len();
            debug_assert!($c.len() == n, "invalid argument c: c.len() = {} != b.len() = {}", $c.len(), n);
            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($b.chunks_exact(8), $c.chunks_exact_mut(8)).for_each(|(b, c)| {
                        $f(&$self, $a, &b[0], &mut c[0]);
                        $f(&$self, $a, &b[1], &mut c[1]);
                        $f(&$self, $a, &b[2], &mut c[2]);
                        $f(&$self, $a, &b[3], &mut c[3]);
                        $f(&$self, $a, &b[4], &mut c[4]);
                        $f(&$self, $a, &b[5], &mut c[5]);
                        $f(&$self, $a, &b[6], &mut c[6]);
                        $f(&$self, $a, &b[7], &mut c[7]);
                    });

                    let m = n - (n&7);
                    izip!($b[m..].iter(), $c[m..].iter_mut()).for_each(|(b, c)| {
                        $f(&$self, $a, b, c);
                    });
                },
                _=>{
                    izip!($b.iter(), $c.iter_mut()).for_each(|(b, c)| {
                        $f(&$self, $a, b, c);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vvsv {

        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $d:expr, $CHUNK:expr) => {

            let n: usize = $a.len();
            debug_assert!($b.len() == n, "invalid argument b: b.len() = {} != a.len() = {}", $b.len(), n);
            debug_assert!($d.len() == n, "invalid argument d: d.len() = {} != a.len() = {}", $d.len(), n);
            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($a.chunks_exact(8), $b.chunks_exact(8), $d.chunks_exact_mut(8)).for_each(|(a, b, d)| {
                        $f(&$self, &a[0], &b[0], $c, &mut d[0]);
                        $f(&$self, &a[1], &b[1], $c, &mut d[1]);
                        $f(&$self, &a[2], &b[2], $c, &mut d[2]);
                        $f(&$self, &a[3], &b[3], $c, &mut d[3]);
                        $f(&$self, &a[4], &b[4], $c, &mut d[4]);
                        $f(&$self, &a[5], &b[5], $c, &mut d[5]);
                        $f(&$self, &a[6], &b[6], $c, &mut d[6]);
                        $f(&$self, &a[7], &b[7], $c, &mut d[7]);
                    });

                    let m = n - (n&7);
                    izip!($a[m..].iter(), $b[m..].iter(), $d[m..].iter_mut()).for_each(|(a, b, d)| {
                        $f(&$self, a, b, $c, d);
                    });
                },
                _=>{
                    izip!($a.iter(), $b.iter(), $d.iter_mut()).for_each(|(a, b, d)| {
                        $f(&$self, a, b, $c, d);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vsv {

        ($self:expr, $f:expr, $a:expr, $c:expr, $b:expr, $CHUNK:expr) => {

            let n: usize = $a.len();
            debug_assert!($b.len() == n, "invalid argument b: b.len() = {} != a.len() = {}", $b.len(), n);
            debug_assert!(CHUNK&(CHUNK-1) == 0, "invalid CHUNK const: not a power of two");

            match CHUNK{
                8 => {
                    
                    izip!($a.chunks_exact(8), $b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        $f(&$self, &a[0], $c, &mut b[0]);
                        $f(&$self, &a[1], $c, &mut b[1]);
                        $f(&$self, &a[2], $c, &mut b[2]);
                        $f(&$self, &a[3], $c, &mut b[3]);
                        $f(&$self, &a[4], $c, &mut b[4]);
                        $f(&$self, &a[5], $c, &mut b[5]);
                        $f(&$self, &a[6], $c, &mut b[6]);
                        $f(&$self, &a[7], $c, &mut b[7]);
                    });

                    let m = n - (n&7);
                    izip!($a[m..].iter(), $b[m..].iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, $c, b);
                    });
                },
                _=>{
                    izip!($a.iter(), $b.iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, $c, b);
                    });
                }
            }
        };
    }
}