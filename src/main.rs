use lib_matrix::matrix;
use lib_nn::{Activation, NeuralNetwork};

fn main() {
    let m = matrix![
        0.2, 0.3, 0.4, 0.5;
    ];
    println!("{m}");

    let n = NeuralNetwork::new(&[4, 6, 7, 8, 2], Activation::SIGMOID);
    let o = n.forward_propagation(&m);
    println!("{o}");
}
