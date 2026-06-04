use std::path::{Path, PathBuf};

use nn::{Activation, LossFunction, Matrix, NeuralNetwork, Optimization, matrix};

use anyhow::Result;

const INPUTS: &[Matrix] = &[
    matrix![0, 0], // 0 + 0
    matrix![0, 1], // 0 + 1
    matrix![1, 0], // 1 + 0
    matrix![1, 1], // 1 + 1
];

const OUTPUTS: &[Matrix] = &[
    //      0, 1, 2
    matrix![1, 0, 0],
    matrix![0, 1, 0],
    matrix![0, 1, 0],
    matrix![0, 0, 1],
];

fn main() -> Result<()> {
    let p = get_saved(std::env::current_exe()?.parent().unwrap());
    let mut nn = nn_new(&p)?;
    nn.train(10000, &INPUTS, &OUTPUTS);
    nn_test(&mut nn);
    nn.save(p)?;

    Ok(())
}

pub fn nn_new(saved: impl AsRef<Path>) -> Result<NeuralNetwork> {
    let nn = if !saved.as_ref().exists() {
        println!("Found no saved neural network in executable dir");
        NeuralNetwork::new(
            [
                (2, Activation::Softplus), // input size
                (8, Activation::Softplus), // hidden layer 1
                (3, Activation::Softmax),  // output size
            ],
            LossFunction::MeanSquaredError,
            // Optimization::NONE,
            Optimization::rmsprop(0.01, 0.9, 1e-8),
        )
    } else {
        println!("Using saved neural network");
        NeuralNetwork::load(&saved)?
    };

    Ok(nn)
}

pub fn nn_test(nn: &mut NeuralNetwork) {
    for (input, output) in (*INPUTS).iter().zip(&*OUTPUTS) {
        let guess = nn.guess(&input);
        let loss = LossFunction::MeanSquaredError.loss(&guess, output);
        println!("Input: {input}Guess: {guess}Correct: {output}Loss: {loss:7.5}\n");
    }
}

fn get_saved(parent_dir: impl AsRef<Path>) -> PathBuf {
    parent_dir.as_ref().join("nn-addition-1.nn")
}
