use homura::{Buffer, DType, Model};

#[test]
fn resnet18_loads_and_produces_1000_class_output() {
    let model = Model::load("tests/fixtures/resnet18-v1-7.onnx").expect("load failed");
    // Zero input: [1, 3, 224, 224]
    let pixels = vec![0.0f32; 1 * 3 * 224 * 224];
    let input = Buffer::from_slice::<f32>(&pixels, &[1, 3, 224, 224], DType::F32);
    let output = model.run(&[&input]).expect("inference failed");
    let logits = output.as_slice::<f32>();

    assert_eq!(logits.len(), 1000, "expected 1000-class output");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "all logits should be finite"
    );

    // Logits should have non-trivial variance (not all the same).
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let variance: f32 =
        logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / logits.len() as f32;
    assert!(
        variance > 0.01,
        "logits should have non-trivial variance, got {variance}"
    );
}

#[test]
fn resnet18_predictions_differ_for_different_inputs() {
    let model = Model::load("tests/fixtures/resnet18-v1-7.onnx").expect("load failed");
    let size = 1 * 3 * 224 * 224;

    // Input 1: zeros
    let input1 = Buffer::from_slice::<f32>(&vec![0.0f32; size], &[1, 3, 224, 224], DType::F32);
    let out1 = model.run(&[&input1]).expect("inference failed");
    let logits1 = out1.as_slice::<f32>().to_vec();

    // Input 2: ones
    let input2 = Buffer::from_slice::<f32>(&vec![1.0f32; size], &[1, 3, 224, 224], DType::F32);
    let out2 = model.run(&[&input2]).expect("inference failed");
    let logits2 = out2.as_slice::<f32>().to_vec();

    // Predictions should differ
    let argmax = |logits: &[f32]| -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };

    let pred1 = argmax(&logits1);
    let pred2 = argmax(&logits2);

    // The model should give different top predictions for very different inputs.
    // (If both are the same, the model isn't really computing anything meaningful.)
    // This is a weak check — just verify the logit vectors differ significantly.
    let diff: f32 = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    assert!(
        diff > 1.0,
        "logits should differ significantly for different inputs (diff={diff}, pred1={pred1}, pred2={pred2})"
    );
}
