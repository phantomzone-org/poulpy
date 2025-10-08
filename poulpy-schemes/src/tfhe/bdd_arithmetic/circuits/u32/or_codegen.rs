use crate::tfhe::bdd_arithmetic::{BitCircuit, BitCircuitInfo, Circuit, GetBitCircuitInfo, Node};
pub(crate) enum AnyBitCircuit {
    B0(BitCircuit<3, 2>),
    B1(BitCircuit<3, 2>),
    B2(BitCircuit<3, 2>),
    B3(BitCircuit<3, 2>),
    B4(BitCircuit<3, 2>),
    B5(BitCircuit<3, 2>),
    B6(BitCircuit<3, 2>),
    B7(BitCircuit<3, 2>),
    B8(BitCircuit<3, 2>),
    B9(BitCircuit<3, 2>),
    B10(BitCircuit<3, 2>),
    B11(BitCircuit<3, 2>),
    B12(BitCircuit<3, 2>),
    B13(BitCircuit<3, 2>),
    B14(BitCircuit<3, 2>),
    B15(BitCircuit<3, 2>),
    B16(BitCircuit<3, 2>),
    B17(BitCircuit<3, 2>),
    B18(BitCircuit<3, 2>),
    B19(BitCircuit<3, 2>),
    B20(BitCircuit<3, 2>),
    B21(BitCircuit<3, 2>),
    B22(BitCircuit<3, 2>),
    B23(BitCircuit<3, 2>),
    B24(BitCircuit<3, 2>),
    B25(BitCircuit<3, 2>),
    B26(BitCircuit<3, 2>),
    B27(BitCircuit<3, 2>),
    B28(BitCircuit<3, 2>),
    B29(BitCircuit<3, 2>),
    B30(BitCircuit<3, 2>),
    B31(BitCircuit<3, 2>),
}
impl BitCircuitInfo for AnyBitCircuit {
    fn info(&self) -> (&[Node], &[usize], usize) {
        match self {
            AnyBitCircuit::B0(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B1(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B2(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B3(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B4(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B5(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B6(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B7(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B8(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B9(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B10(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B11(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B12(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B13(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B14(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B15(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B16(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B17(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B18(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B19(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B20(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B21(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B22(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B23(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B24(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B25(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B26(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B27(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B28(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B29(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B30(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
            AnyBitCircuit::B31(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
        }
    }
}

impl GetBitCircuitInfo<u32> for Circuit<AnyBitCircuit, 32usize> {
    fn input_size(&self) -> usize {
        2 * u32::BITS as usize
    }
    fn output_size(&self) -> usize {
        u32::BITS as usize
    }
    fn get_circuit(&self, bit: usize) -> (&[Node], &[usize], usize) {
        self.0[bit].info()
    }
}

pub(crate) static OUTPUT_CIRCUITS: Circuit<AnyBitCircuit, 32usize> = Circuit([
    AnyBitCircuit::B0(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(32, 1, 0), Node::new(0, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B1(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(33, 1, 0), Node::new(1, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B2(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(34, 1, 0), Node::new(2, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B3(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(35, 1, 0), Node::new(3, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B4(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(36, 1, 0), Node::new(4, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B5(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(37, 1, 0), Node::new(5, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B6(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(38, 1, 0), Node::new(6, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B7(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(39, 1, 0), Node::new(7, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B8(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(40, 1, 0), Node::new(8, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B9(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(41, 1, 0), Node::new(9, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B10(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(42, 1, 0), Node::new(10, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B11(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(43, 1, 0), Node::new(11, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B12(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(44, 1, 0), Node::new(12, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B13(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(45, 1, 0), Node::new(13, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B14(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(46, 1, 0), Node::new(14, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B15(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(47, 1, 0), Node::new(15, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B16(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(48, 1, 0), Node::new(16, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B17(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(49, 1, 0), Node::new(17, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B18(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(50, 1, 0), Node::new(18, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B19(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(51, 1, 0), Node::new(19, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B20(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(52, 1, 0), Node::new(20, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B21(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(53, 1, 0), Node::new(21, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B22(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(54, 1, 0), Node::new(22, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B23(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(55, 1, 0), Node::new(23, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B24(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(56, 1, 0), Node::new(24, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B25(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(57, 1, 0), Node::new(25, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B26(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(58, 1, 0), Node::new(26, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B27(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(59, 1, 0), Node::new(27, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B28(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(60, 1, 0), Node::new(28, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B29(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(61, 1, 0), Node::new(29, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B30(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(62, 1, 0), Node::new(30, 1, 1)],
        [0, 2],
        2,
    )),
    AnyBitCircuit::B31(BitCircuit::new(
        [Node::new(0, 0, 0), Node::new(63, 1, 0), Node::new(31, 1, 1)],
        [0, 2],
        2,
    )),
]);
