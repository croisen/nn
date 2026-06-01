use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use lib_matrix::{Matrix, matrix};
use lib_nn::{Activation, LossFunction, NeuralNetwork, Optimization};

static INPUTS: LazyLock<Vec<Matrix>> =
    LazyLock::new(|| vec![matrix![0, 0], matrix![0, 1], matrix![1, 0], matrix![1, 1]]);
static OUTPUTS: LazyLock<Vec<Matrix>> = LazyLock::new(|| {
    vec![
        matrix![0, 0, 1],
        matrix![0, 1, 0],
        matrix![0, 1, 0],
        matrix![1, 0, 0],
    ]
});

pub fn nn_new(dir: impl AsRef<Path>) -> Result<NeuralNetwork> {
    let saved = get_saved(dir);

    let nn = if !saved.exists() {
        println!("Found no saved neural network in executable dir");
        NeuralNetwork::new(
            [
                (2, Activation::Sigmoid),
                (32, Activation::Sigmoid),
                (3, Activation::Sigmoid),
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

pub fn nn_train(nn: &mut NeuralNetwork) {
    nn.train_until(0.0001, &*INPUTS, &*OUTPUTS);
}

pub fn nn_test(nn: &mut NeuralNetwork) {
    for (input, output) in (*INPUTS).iter().zip(&*OUTPUTS) {
        let guess = nn.guess(&input);
        let loss = LossFunction::MeanSquaredError.loss(&guess, output);
        println!("Input: {input}Guess: {guess}Correct: {output}Loss: {loss:7.5}\n");
    }
}

fn get_saved(parent_dir: impl AsRef<Path>) -> PathBuf {
    parent_dir.as_ref().join("nn-addition.nn")
}
