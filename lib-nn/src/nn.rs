use std::fs::{read as fread, write as fwrite};
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use anyhow::Result;
use flate2::Compression;
use flate2::read::{ZlibDecoder, ZlibEncoder};
use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_string_pretty};

use crate::activation::Activation;
use crate::loss::LossFunction;
use crate::optimization::Optimization;
use lib_matrix::Matrix;

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct NeuralNetwork {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation: Vec<Activation>,
    loss_function: LossFunction,
    optimization: Optimization,
    loss: f64,
}

impl NeuralNetwork {
    pub fn new(
        layers: impl AsRef<[(usize, Activation)]>,
        loss_function: LossFunction,
        optimization: Optimization,
    ) -> Self {
        let (layers, activation): (Vec<usize>, Vec<Activation>) =
            layers.as_ref().into_iter().map(|(a, b)| (*a, *b)).unzip();

        let weights: Vec<Matrix> = layers
            .windows(2)
            .map(|a| Matrix::rand(a[0], a[1])) // i rows, x cols
            .collect();

        let biases: Vec<Matrix> = layers
            .windows(2)
            .map(|a| Matrix::rand(1, a[1])) // 1 row, x cols
            .collect();

        Self {
            activation,
            loss_function,
            optimization,
            weights,
            biases,
            loss: 1.0,
        }
    }

    pub fn save(&self, file: impl AsRef<Path>) -> Result<()> {
        let json = to_string_pretty(self)?;
        let mut e = ZlibEncoder::new(json.as_bytes(), Compression::new(9));
        let mut compressed = vec![];
        e.read_to_end(&mut compressed)?;
        fwrite(file, compressed)?;
        Ok(())
    }

    pub fn load(file: impl AsRef<Path>) -> Result<Self> {
        let json = fread(file)?;
        let mut d = ZlibDecoder::new(json.as_slice());
        let mut decompressed = vec![];
        d.read_to_end(&mut decompressed)?;
        let nn = from_slice(decompressed.as_slice())?;
        Ok(nn)
    }

    pub fn train(
        &mut self,
        epochs: usize,
        inputs: impl AsRef<Vec<Matrix>>,
        correct: impl AsRef<Vec<Matrix>>,
    ) {
        println!("Training neural network until epoch {epochs}");
        let thresh = (epochs / 100).max(1);
        let mut now = SystemTime::now();
        self.optimization.init_cache(&self.weights, &self.biases);
        for epoch in 1..=epochs {
            let mut losses = Box::<[f64]>::new_zeroed_slice(inputs.as_ref().len());
            for (i, input) in inputs.as_ref().iter().enumerate() {
                let (pre, post) = self.forward_propagation(input);
                let loss = self.backward_propagation(pre, post, &correct.as_ref()[i]);
                losses[i].write(loss);
            }

            if epoch % (thresh * 10) == 0 {
                let loss = unsafe {
                    losses.iter().map(|v| v.assume_init()).sum::<f64>() / losses.len() as f64
                };

                let n = SystemTime::now();
                let elapsed = n.duration_since(now).unwrap();
                now = n;
                self.loss = loss;
                println!(
                    "Epoch {epoch}/{epochs} Loss: {loss:08.6} | Elapsed: {}ms",
                    elapsed.as_millis()
                );
            } else if epoch % thresh == 0 {
                println!("Epoch {epoch}/{epochs}");
            }
        }
    }

    pub fn train_until(
        &mut self,
        loss: f64,
        inputs: impl AsRef<Vec<Matrix>>,
        correct: impl AsRef<Vec<Matrix>>,
    ) {
        println!("Training neural network until loss is less than: {loss}");
        self.optimization.init_cache(&self.weights, &self.biases);
        let mut now = SystemTime::now();
        let mut force_break = 0;
        for epoch in 1.. {
            let mut losses = Box::<[f64]>::new_zeroed_slice(inputs.as_ref().len());
            for (i, input) in inputs.as_ref().iter().enumerate() {
                let (pre, post) = self.forward_propagation(input);
                let loss = self.backward_propagation(pre, post, &correct.as_ref()[i]);
                losses[i].write(loss);
            }

            let avg_loss = unsafe {
                losses.iter().map(|v| v.assume_init()).sum::<f64>() / losses.len() as f64
            };

            if avg_loss <= loss {
                break;
            }

            if epoch % 100 == 0 {
                let n = SystemTime::now();
                let elapsed = n.duration_since(now).unwrap();
                now = n;

                println!(
                    "Epoch {epoch}/... Loss: {avg_loss:08.6} | Elapsed: {}ms",
                    elapsed.as_millis()
                );
                if self.loss - avg_loss <= 0.0000001 {
                    println!(
                        "Loss is not going down\n\tprev: {} | now: {avg_loss}",
                        self.loss
                    );

                    force_break += 1;
                }

                if force_break == 10 {
                    break;
                }
            }

            self.loss = avg_loss;
        }
    }

    pub fn guess(&mut self, input: impl AsRef<Matrix>) -> Matrix {
        let mut m = input.as_ref().to_owned();
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            m = self.activation[i].activate(m * w + b);
        }

        m
    }

    fn forward_propagation(&self, input: impl AsRef<Matrix>) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut m = input.as_ref().to_owned();
        let mut pre = Vec::with_capacity(self.weights.len());
        let mut post = Vec::with_capacity(self.weights.len() + 1);
        post.push(m.clone());
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let tmp = m * w + b;
            pre.push(tmp.clone());
            m = self.activation[i].activate(tmp);
            post.push(m.clone())
        }

        (pre, post)
    }

    fn backward_propagation(
        &mut self,
        pre: Vec<Matrix>,
        post: Vec<Matrix>,
        correct: impl AsRef<Matrix>,
    ) -> f64 {
        let output = post.last().unwrap();
        let loss = self.loss_function.loss(output, &correct);
        let mut error = self.loss_function.derivative(output, correct);
        for i in (0..post.len() - 1).rev() {
            let activated = error.hadamard_mul(&self.activation[i].derivative(&pre[i]));
            let weight_grad = post[i].transpose() * &activated;
            let bias_grad = activated.sum_over_rows();
            error = activated * self.weights[i].transpose();
            let (wg, bg) = self.optimization.optimize(i, &weight_grad, &bias_grad);
            self.weights[i] -= wg;
            self.biases[i] -= bg;
        }

        loss
    }
}
