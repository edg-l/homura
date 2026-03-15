use homura::{Buffer, DType, Model};

fn main() {
    let model = Model::load("tests/fixtures/mnist-12.onnx").expect("failed to load mnist-12.onnx");

    // Blank image (all zeros) — model should predict a digit
    let input = Buffer::from_slice::<f32>(&vec![0.0f32; 784], &[1, 1, 28, 28], DType::F32);
    let output = model.run(&[&input]).expect("inference failed");
    let logits = output.as_slice::<f32>();

    println!("MNIST logits (blank input):");
    for (digit, &logit) in logits.iter().enumerate() {
        let bar = "|".repeat(((logit + 10.0).max(0.0) * 2.0) as usize);
        println!("  {digit}: {logit:>8.3}  {bar}");
    }

    let predicted = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("\nPredicted digit: {predicted}");
}
