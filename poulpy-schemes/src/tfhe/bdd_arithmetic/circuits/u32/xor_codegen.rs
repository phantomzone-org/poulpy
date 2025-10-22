use crate::tfhe::bdd_arithmetic::{BitCircuit, BitCircuitInfo, Circuit, GetBitCircuitInfo, Node};
pub(crate) enum AnyBitCircuit {
    B0(BitCircuit<4>),
    B1(BitCircuit<4>),
    B2(BitCircuit<4>),
    B3(BitCircuit<4>),
    B4(BitCircuit<4>),
    B5(BitCircuit<4>),
    B6(BitCircuit<4>),
    B7(BitCircuit<4>),
    B8(BitCircuit<4>),
    B9(BitCircuit<4>),
    B10(BitCircuit<4>),
    B11(BitCircuit<4>),
    B12(BitCircuit<4>),
    B13(BitCircuit<4>),
    B14(BitCircuit<4>),
    B15(BitCircuit<4>),
    B16(BitCircuit<4>),
    B17(BitCircuit<4>),
    B18(BitCircuit<4>),
    B19(BitCircuit<4>),
    B20(BitCircuit<4>),
    B21(BitCircuit<4>),
    B22(BitCircuit<4>),
    B23(BitCircuit<4>),
    B24(BitCircuit<4>),
    B25(BitCircuit<4>),
    B26(BitCircuit<4>),
    B27(BitCircuit<4>),
    B28(BitCircuit<4>),
    B29(BitCircuit<4>),
    B30(BitCircuit<4>),
    B31(BitCircuit<4>),
}
impl BitCircuitInfo for AnyBitCircuit {
    fn info(&self) -> (&[Node], usize) {
        match self {
            AnyBitCircuit::B0(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B1(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B2(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B3(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B4(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B5(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B6(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B7(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B8(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B9(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B10(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B11(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B12(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B13(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B14(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B15(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B16(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B17(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B18(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B19(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B20(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B21(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B22(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B23(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B24(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B25(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B26(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B27(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B28(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B29(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B30(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
            AnyBitCircuit::B31(bit_circuit) => (bit_circuit.nodes.as_ref(), bit_circuit.max_inter_state),
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
    fn get_circuit(&self, bit: usize) -> (&[Node], usize) {
        self.0[bit].info()
    }
}

pub(crate) static OUTPUT_CIRCUITS: Circuit<AnyBitCircuit, 32usize> = Circuit([
    AnyBitCircuit::B0(BitCircuit::new(
        [
            Node::Cmux(32, 0, 1),
            Node::Cmux(32, 1, 0),
            Node::Cmux(0, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B1(BitCircuit::new(
        [
            Node::Cmux(33, 0, 1),
            Node::Cmux(33, 1, 0),
            Node::Cmux(1, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B2(BitCircuit::new(
        [
            Node::Cmux(34, 1, 0),
            Node::Cmux(34, 0, 1),
            Node::Cmux(2, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B3(BitCircuit::new(
        [
            Node::Cmux(35, 0, 1),
            Node::Cmux(35, 1, 0),
            Node::Cmux(3, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B4(BitCircuit::new(
        [
            Node::Cmux(36, 0, 1),
            Node::Cmux(36, 1, 0),
            Node::Cmux(4, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B5(BitCircuit::new(
        [
            Node::Cmux(37, 0, 1),
            Node::Cmux(37, 1, 0),
            Node::Cmux(5, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B6(BitCircuit::new(
        [
            Node::Cmux(38, 1, 0),
            Node::Cmux(38, 0, 1),
            Node::Cmux(6, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B7(BitCircuit::new(
        [
            Node::Cmux(39, 1, 0),
            Node::Cmux(39, 0, 1),
            Node::Cmux(7, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B8(BitCircuit::new(
        [
            Node::Cmux(40, 0, 1),
            Node::Cmux(40, 1, 0),
            Node::Cmux(8, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B9(BitCircuit::new(
        [
            Node::Cmux(41, 1, 0),
            Node::Cmux(41, 0, 1),
            Node::Cmux(9, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B10(BitCircuit::new(
        [
            Node::Cmux(42, 0, 1),
            Node::Cmux(42, 1, 0),
            Node::Cmux(10, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B11(BitCircuit::new(
        [
            Node::Cmux(43, 0, 1),
            Node::Cmux(43, 1, 0),
            Node::Cmux(11, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B12(BitCircuit::new(
        [
            Node::Cmux(44, 0, 1),
            Node::Cmux(44, 1, 0),
            Node::Cmux(12, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B13(BitCircuit::new(
        [
            Node::Cmux(45, 1, 0),
            Node::Cmux(45, 0, 1),
            Node::Cmux(13, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B14(BitCircuit::new(
        [
            Node::Cmux(46, 1, 0),
            Node::Cmux(46, 0, 1),
            Node::Cmux(14, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B15(BitCircuit::new(
        [
            Node::Cmux(47, 1, 0),
            Node::Cmux(47, 0, 1),
            Node::Cmux(15, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B16(BitCircuit::new(
        [
            Node::Cmux(48, 0, 1),
            Node::Cmux(48, 1, 0),
            Node::Cmux(16, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B17(BitCircuit::new(
        [
            Node::Cmux(49, 0, 1),
            Node::Cmux(49, 1, 0),
            Node::Cmux(17, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B18(BitCircuit::new(
        [
            Node::Cmux(50, 0, 1),
            Node::Cmux(50, 1, 0),
            Node::Cmux(18, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B19(BitCircuit::new(
        [
            Node::Cmux(51, 1, 0),
            Node::Cmux(51, 0, 1),
            Node::Cmux(19, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B20(BitCircuit::new(
        [
            Node::Cmux(52, 1, 0),
            Node::Cmux(52, 0, 1),
            Node::Cmux(20, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B21(BitCircuit::new(
        [
            Node::Cmux(53, 0, 1),
            Node::Cmux(53, 1, 0),
            Node::Cmux(21, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B22(BitCircuit::new(
        [
            Node::Cmux(54, 1, 0),
            Node::Cmux(54, 0, 1),
            Node::Cmux(22, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B23(BitCircuit::new(
        [
            Node::Cmux(55, 1, 0),
            Node::Cmux(55, 0, 1),
            Node::Cmux(23, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B24(BitCircuit::new(
        [
            Node::Cmux(56, 0, 1),
            Node::Cmux(56, 1, 0),
            Node::Cmux(24, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B25(BitCircuit::new(
        [
            Node::Cmux(57, 0, 1),
            Node::Cmux(57, 1, 0),
            Node::Cmux(25, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B26(BitCircuit::new(
        [
            Node::Cmux(58, 1, 0),
            Node::Cmux(58, 0, 1),
            Node::Cmux(26, 1, 0),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B27(BitCircuit::new(
        [
            Node::Cmux(59, 0, 1),
            Node::Cmux(59, 1, 0),
            Node::Cmux(27, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B28(BitCircuit::new(
        [
            Node::Cmux(60, 0, 1),
            Node::Cmux(60, 1, 0),
            Node::Cmux(28, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B29(BitCircuit::new(
        [
            Node::Cmux(61, 0, 1),
            Node::Cmux(61, 1, 0),
            Node::Cmux(29, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B30(BitCircuit::new(
        [
            Node::Cmux(62, 0, 1),
            Node::Cmux(62, 1, 0),
            Node::Cmux(30, 0, 1),
            Node::None,
        ],
        2,
    )),
    AnyBitCircuit::B31(BitCircuit::new(
        [
            Node::Cmux(63, 1, 0),
            Node::Cmux(63, 0, 1),
            Node::Cmux(31, 1, 0),
            Node::None,
        ],
        2,
    )),
]);
