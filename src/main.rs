use anyhow::Result;

use lib_matrix::matrix;
use lib_nn::{Activation, LossFunction, NeuralNetwork};

fn main() -> Result<()> {
    let inputs = vec![matrix![0, 0], matrix![0, 1], matrix![1, 0], matrix![1, 1]];
    let outputs = vec![matrix![0, 0], matrix![0, 1], matrix![0, 1], matrix![1, 0]];
    let mut nn = NeuralNetwork::new(
        [
            (2, Activation::Sigmoid),
            (5, Activation::Sigmoid),
            (2, Activation::Sigmoid),
        ],
        LossFunction::MeanSquaredError,
        0.01,
    );

    nn.train(1000, &inputs, &outputs);
    for (input, output) in inputs.iter().zip(&outputs) {
        let guess = nn.guess(&input);
        println!("Input: {input}Guess: {guess}Correct: {output}");
    }

    Ok(())
}
