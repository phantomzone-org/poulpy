use crate::tfhe::bdd_arithmetic::{BitCircuit, BitCircuitInfo, Circuit, GetBitCircuitInfo, Node};
pub(crate) enum AnyBitCircuit {
    B0(BitCircuit<3, 2>),
}

impl BitCircuitInfo for AnyBitCircuit {
    fn info(&self) -> (&[Node], &[usize], usize) {
        match self {
            AnyBitCircuit::B0(bit_circuit) => (
                bit_circuit.nodes.as_ref(),
                bit_circuit.levels.as_ref(),
                bit_circuit.max_inter_state,
            ),
        }
    }
}

impl GetBitCircuitInfo<u32> for Circuit<AnyBitCircuit, 1usize> {
    fn input_size(&self) -> usize {
        2 * u32::BITS as usize
    }
    fn output_size(&self) -> usize {
        u32::BITS as usize
    }
    fn get_circuit(&self, _bit: usize) -> (&[Node], &[usize], usize) {
        self.0[0].info()
    }
}

pub(crate) static OUTPUT_CIRCUITS: Circuit<AnyBitCircuit, 1usize> = Circuit([AnyBitCircuit::B0(BitCircuit::new(
    [Node::new(1, 1, 0), Node::new(1, 0, 1), Node::new(0, 1, 0)],
    [0, 2],
    2,
))]);
