use homura::{Compiler, DType, Tensor, begin_trace, take_trace};

fn main() {
    println!("homura: a + b demo");

    // Record the trace.
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = Tensor::new(&[4], DType::F32);
    let c = &a + &b;
    let trace = take_trace();

    println!("Trace has {} op(s)", trace.ops().len());

    // Compile.
    let compiled = Compiler::compile(&trace, &[c.id()]).expect("compilation failed");

    // Execute.
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];
    let result = compiled.run(&[&a_data, &b_data]);

    println!("a      = {:?}", a_data);
    println!("b      = {:?}", b_data);
    println!("a + b  = {:?}", result);

    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    println!("Result verified.");

    // Demonstrate chained add: a + b + c
    begin_trace();
    let x = Tensor::new(&[4], DType::F32);
    let y = Tensor::new(&[4], DType::F32);
    let z = Tensor::new(&[4], DType::F32);
    let out = &(&x + &y) + &z;
    let trace2 = take_trace();

    let compiled2 = Compiler::compile(&trace2, &[out.id()]).expect("chained compile failed");
    let result2 = compiled2.run(&[&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], &[100.0, 200.0, 300.0, 400.0]]);
    println!("x + y + z = {:?}", result2);
    assert_eq!(result2, vec![111.0, 222.0, 333.0, 444.0]);
    println!("Chained result verified.");
}
