use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use nn::{Activation, LossFunction, Matrix, NeuralNetwork, Optimization, matrix};

static INPUTS: LazyLock<Vec<Matrix>> = LazyLock::new(|| {
    vec![
        matrix![0, 0], // 0 xor 0
        matrix![0, 1], // 0 xor 1
        matrix![1, 0], // 1 xor 0
        matrix![1, 1], // 1 xor 1
    ]
});
static OUTPUTS: LazyLock<Vec<Matrix>> = LazyLock::new(|| {
    vec![
        //      0, 1
        matrix![1, 0],
        matrix![0, 1],
        matrix![0, 1],
        matrix![1, 0],
    ]
});

fn main() -> Result<()> {
    let p = get_saved(std::env::current_exe()?.parent().unwrap());
    let mut nn = nn_new(&p)?;
    nn.train(1000, &*INPUTS, &*OUTPUTS);
    nn_test(&mut nn);
    nn.save(p)?;

    Ok(())
}

fn nn_new(saved: impl AsRef<Path>) -> Result<NeuralNetwork> {
    let nn = if !saved.as_ref().exists() {
        println!("Found no saved neural network in executable dir");
        let mut nn = NeuralNetwork::new(
            [
                (2, Activation::Sigmoid),  // input size
                (16, Activation::Sigmoid), // hidden layer 1
                (2, Activation::Softmax),  // output size
            ],
            LossFunction::MeanSquaredError,
            // Optimization::NONE,
            Optimization::rmsprop(0.01, 0.9, 1e-8),
        );

        nn.train_until(0.00001, &*INPUTS, &*OUTPUTS);
        nn
    } else {
        println!("Using saved neural network");
        NeuralNetwork::load(&saved)?
    };

    Ok(nn)
}

fn nn_test(nn: &mut NeuralNetwork) {
    for (input, output) in (*INPUTS).iter().zip(&*OUTPUTS) {
        let guess = nn.guess(&input);
        let loss = LossFunction::MeanSquaredError.loss(&guess, output);
        println!("Input: {input}Guess: {guess}Correct: {output}Loss: {loss:7.5}\n");
    }
}

fn get_saved(parent_dir: impl AsRef<Path>) -> PathBuf {
    parent_dir.as_ref().join("nn-xor.nn")
}
