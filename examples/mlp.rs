use homura::{Buffer, DType, Tensor, begin_trace};

fn main() {
    println!("homura: hand-coded MLP demo");
    println!();

    // 2-layer MLP: [batch, 4] -> [batch, 8] -> [batch, 3]
    // Using small dimensions for demo purposes.
    let batch = 2;
    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 3;

    begin_trace();
    let x = Tensor::new(&[batch, input_dim], DType::F32);
    let w1 = Tensor::new(&[input_dim, hidden_dim], DType::F32);
    let b1 = Tensor::new(&[hidden_dim], DType::F32);
    let w2 = Tensor::new(&[hidden_dim, output_dim], DType::F32);
    let b2 = Tensor::new(&[output_dim], DType::F32);

    // Layer 1: relu(x @ w1 + b1)
    let h = (x.matmul(&w1) + &b1).relu();
    // Layer 2: softmax(h @ w2 + b2)
    let out = (h.matmul(&w2) + &b2).softmax(-1);

    // Create input data
    // x: 2x4 input
    let x_data: Vec<f32> = (0..batch * input_dim).map(|i| (i as f32) * 0.1).collect();
    // w1: 4x8 weights (small random-ish values)
    let w1_data: Vec<f32> = (0..input_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.13 - 0.5) * 0.5)
        .collect();
    // b1: 8 biases
    let b1_data: Vec<f32> = vec![0.0; hidden_dim as usize];
    // w2: 8x3 weights
    let w2_data: Vec<f32> = (0..hidden_dim * output_dim)
        .map(|i| ((i as f32) * 0.17 - 0.5) * 0.5)
        .collect();
    // b2: 3 biases
    let b2_data: Vec<f32> = vec![0.0; output_dim as usize];

    let x_buf = Buffer::from_slice::<f32>(&x_data, &[batch, input_dim], DType::F32);
    let w1_buf = Buffer::from_slice::<f32>(&w1_data, &[input_dim, hidden_dim], DType::F32);
    let b1_buf = Buffer::from_slice::<f32>(&b1_data, &[hidden_dim], DType::F32);
    let w2_buf = Buffer::from_slice::<f32>(&w2_data, &[hidden_dim, output_dim], DType::F32);
    let b2_buf = Buffer::from_slice::<f32>(&b2_data, &[output_dim], DType::F32);

    let result = out.eval(&[x_buf, w1_buf, b1_buf, w2_buf, b2_buf]);

    println!("Input ({}x{}):", batch, input_dim);
    for row in 0..batch as usize {
        let start = row * input_dim as usize;
        let end = start + input_dim as usize;
        println!("  {:?}", &x_data[start..end]);
    }
    println!();

    println!("Output ({}x{}) — softmax probabilities:", batch, output_dim);
    let out_data = result.as_slice::<f32>();
    for row in 0..batch as usize {
        let start = row * output_dim as usize;
        let end = start + output_dim as usize;
        let row_data = &out_data[start..end];
        let row_sum: f32 = row_data.iter().sum();
        println!("  {:?} (sum: {:.6})", row_data, row_sum);
    }
    println!();

    // Verify softmax properties
    for row in 0..batch as usize {
        let start = row * output_dim as usize;
        let end = start + output_dim as usize;
        let row_data = &out_data[start..end];
        let row_sum: f32 = row_data.iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "Row {} sum is {}, expected ~1.0",
            row,
            row_sum
        );
        for &v in row_data {
            assert!(v > 0.0 && v < 1.0, "Softmax value {} out of (0, 1) range", v);
        }
    }
    println!("All assertions passed! Softmax outputs are valid probabilities.");
}
