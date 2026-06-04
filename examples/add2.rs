use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use nn::{Activation, LossFunction, Matrix, NeuralNetwork, Optimization, matrix};

use anyhow::Result;

const INPUTS: &[Matrix] = &[
    matrix![0, 0, 0, 0], // 00 + 00 = 0
    matrix![0, 0, 0, 1], // 00 + 01 = 1
    matrix![0, 0, 1, 0], // 00 + 10 = 2
    matrix![0, 0, 1, 1], // 00 + 11 = 3
    matrix![0, 1, 0, 0], // 01 + 00 = 1
    matrix![0, 1, 0, 1], // 01 + 01 = 2
    matrix![0, 1, 1, 0], // 01 + 10 = 3
    matrix![0, 1, 1, 1], // 01 + 11 = 4
    matrix![1, 0, 0, 0], // 10 + 00 = 2
    matrix![1, 0, 0, 1], // 10 + 01 = 3
    matrix![1, 0, 1, 0], // 10 + 10 = 4
    matrix![1, 0, 1, 1], // 10 + 11 = 5
    matrix![1, 1, 0, 0], // 11 + 00 = 3
    matrix![1, 1, 0, 1], // 11 + 01 = 4
    matrix![1, 1, 1, 0], // 11 + 10 = 5
    matrix![1, 1, 1, 1], // 11 + 11 = 6
];

const OUTPUTS: &[Matrix] = &[
    //      0, 1, 2, 3, 4, 5, 6
    matrix![1, 0, 0, 0, 0, 0, 0],
    matrix![1, @ 7], // x = [0 as f64; 7], x[1] = 1.0;
    matrix![2, @ 7],
    matrix![3, @ 7],
    matrix![1, @ 7],
    matrix![2, @ 7],
    matrix![3, @ 7],
    matrix![4, @ 7],
    matrix![2, @ 7],
    matrix![3, @ 7],
    matrix![4, @ 7],
    matrix![5, @ 7],
    matrix![3, @ 7],
    matrix![4, @ 7],
    matrix![5, @ 7],
    matrix![6, @ 7],
];

fn main() -> Result<()> {
    let p = get_saved(std::env::current_exe()?.parent().unwrap());
    let mut nn = nn_new(&p)?;
    nn.train(10000, INPUTS, OUTPUTS);
    nn_test(&mut nn);
    nn.save(p)?;

    Ok(())
}

pub fn nn_new(saved: impl AsRef<Path>) -> Result<NeuralNetwork> {
    let nn = if !saved.as_ref().exists() {
        println!("Found no saved neural network in executable dir");
        let mut nn = NeuralNetwork::new(
            [
                (4, Activation::Sigmoid), // input size
                (8, Activation::Sigmoid), // hidden layer 1
                (7, Activation::Softmax), // output size
            ],
            LossFunction::MeanSquaredError,
            // Optimization::NONE,
            Optimization::rmsprop(0.01, 0.9, 1e-8),
        );
        nn.train_until(0.00001, &*INPUTS, &*OUTPUTS);
        nn
    } else {
        println!("Using saved neural network");
        NeuralNetwork::load(saved)?
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
    parent_dir.as_ref().join("nn-addition-2.nn")
}
