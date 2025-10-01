use crate::tfhe::arithmetic::{BitCircuit, Circuit, Node};

pub(crate) static OUTPUT_CIRCUIT: Circuit<BitCircuit<4usize, 3usize>, 1> = Circuit([BitCircuit::new(
    [
        Node::new(0, 0, 0),
        Node::new(0, 0, 0),
        Node::new(1, 1, 0),
        Node::new(0, 1, 2),
    ],
    [2, 3, 4],
)]);
