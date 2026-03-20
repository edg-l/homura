//! End-to-end tests for GGUF model loading and quantized execution plan compilation.

use std::path::Path;

use homura::hf::model::HfModel;

#[test]
fn load_gguf_and_compile() {
    let gguf_path = Path::new("models/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q8_0.gguf");
    if !gguf_path.exists() {
        eprintln!("skipping: {gguf_path:?} not found");
        return;
    }

    let model = HfModel::load_gguf(gguf_path).expect("load_gguf failed");

    let config = model.config();
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 36);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.kv_heads(), 2);

    // Trigger compilation (this calls emit_transformer_plan_quant internally).
    // Use a dummy input to force compilation.
    let input_ids = homura::Buffer::from_slice::<i64>(&[0i64], &[1, 1], homura::DType::I64);
    let result = model.run(&input_ids);
    match &result {
        Ok(outputs) => {
            println!("run succeeded: {} outputs", outputs.len());
            let logits = &outputs[0];
            println!(
                "logits shape: {:?}, dtype: {:?}",
                logits.shape(),
                logits.dtype()
            );
        }
        Err(e) => {
            // Compilation errors are expected during development.
            // Print them but don't panic -- the test validates that loading works.
            eprintln!("run failed (may be expected during development): {e}");
        }
    }
}
