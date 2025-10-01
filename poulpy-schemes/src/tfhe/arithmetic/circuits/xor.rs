use crate::tfhe::arithmetic::{BitCircuit, Node};

pub(crate) static OUTPUT_CIRCUIT: BitCircuit<5usize, 3usize> = BitCircuit::new(
    [
        Node::new(0, 0, 0),
        Node::new(0, 0, 0),
        Node::new(1, 0, 1),
        Node::new(1, 1, 0),
        Node::new(0, 2, 3),
    ],
    [2, 4, 5],
);
