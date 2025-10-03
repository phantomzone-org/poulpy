use crate::tfhe::bdd_arithmetic::{BitCircuit, BitCircuitInfo, Circuit, Node};
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
pub(crate) static OUTPUT_CIRCUITS: Circuit<AnyBitCircuit, 1usize> = Circuit([AnyBitCircuit::B0(BitCircuit::new(
    [Node::new(0, 0, 0), Node::new(1, 1, 0), Node::new(0, 1, 0)],
    [0, 2],
    2,
))]);
