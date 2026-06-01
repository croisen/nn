use std::env::current_exe;

use anyhow::Result;

mod addition;
mod minist;

fn main() -> Result<()> {
    let saved = current_exe()?.parent().unwrap().to_path_buf();
    println!("Addition Test");
    let mut nn = addition::nn_new(&saved)?;
    addition::nn_train(&mut nn);
    addition::nn_test(&mut nn);

    println!("MNIST Test");
    let mut nn = minist::nn_new(&saved)?;
    minist::nn_train(&mut nn);
    minist::nn_test(&mut nn);

    Ok(())
}
