use base2k::{FFT64, Module, Scratch};

pub trait KeySwitchScratchSpace {
    fn keyswitch_scratch_space(module: &Module<FFT64>, res_size: usize, lhs: usize, rhs: usize) -> usize;
}

pub trait KeySwitch<DataLhs, DataRhs> {
    type Lhs;
    type Rhs;
    fn keyswitch(&mut self, module: &Module<FFT64>, lhs: &Self::Lhs, rhs: &Self::Rhs, scratch: &mut Scratch);
}

pub trait KeySwitchInplaceScratchSpace {
    fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, res_size: usize, rhs: usize) -> usize;
}

pub trait KeySwitchInplace<DataRhs> {
    type Rhs;
    fn keyswitch_inplace(&mut self, module: &Module<FFT64>, rhs: &Self::Rhs, scratch: &mut Scratch);
}
