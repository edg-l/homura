use homura::{Compiler, DType, Tensor, begin_trace, take_trace};

fn main() {
    println!("homura: element-wise ops demo");
    println!();

    // Sub: [10, 20, 30, 40] - [1, 2, 3, 4] = [9, 18, 27, 36]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = Tensor::new(&[4], DType::F32);
    let c = &a - &b;
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[c.id()]).expect("compile failed");
    let result = compiled.run(&[&[10.0, 20.0, 30.0, 40.0], &[1.0, 2.0, 3.0, 4.0]]);
    println!("Sub: [10,20,30,40] - [1,2,3,4] = {:?}", result);
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);

    // Mul: [1, 2, 3, 4] * [5, 6, 7, 8] = [5, 12, 21, 32]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = Tensor::new(&[4], DType::F32);
    let c = &a * &b;
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[c.id()]).expect("compile failed");
    let result = compiled.run(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
    println!("Mul: [1,2,3,4] * [5,6,7,8] = {:?}", result);
    assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);

    // Div: [10, 20, 30, 40] / [2, 4, 5, 8] = [5, 5, 6, 5]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = Tensor::new(&[4], DType::F32);
    let c = &a / &b;
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[c.id()]).expect("compile failed");
    let result = compiled.run(&[&[10.0, 20.0, 30.0, 40.0], &[2.0, 4.0, 5.0, 8.0]]);
    println!("Div: [10,20,30,40] / [2,4,5,8] = {:?}", result);
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);

    // Neg: -[1, -2, 3, -4] = [-1, 2, -3, 4]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = -&a;
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[b.id()]).expect("compile failed");
    let result = compiled.run(&[&[1.0, -2.0, 3.0, -4.0]]);
    println!("Neg: -[1,-2,3,-4] = {:?}", result);
    assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);

    // Relu: relu([-1, 2, -3, 4]) = [0, 2, 0, 4]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = a.relu();
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[b.id()]).expect("compile failed");
    let result = compiled.run(&[&[-1.0, 2.0, -3.0, 4.0]]);
    println!("Relu: relu([-1,2,-3,4]) = {:?}", result);
    assert_eq!(result, vec![0.0, 2.0, 0.0, 4.0]);

    // Chained: relu(a + b) where a=[1,-5,3,-7], b=[2,3,-4,5] -> [3,-2,-1,-2] -> [3,0,0,0]
    begin_trace();
    let a = Tensor::new(&[4], DType::F32);
    let b = Tensor::new(&[4], DType::F32);
    let c = (&a + &b).relu();
    let trace = take_trace();
    let compiled = Compiler::compile(&trace, &[c.id()]).expect("compile failed");
    let result = compiled.run(&[&[1.0, -5.0, 3.0, -7.0], &[2.0, 3.0, -4.0, 5.0]]);
    println!("Chained relu(a+b): relu([3,-2,-1,-2]) = {:?}", result);
    assert_eq!(result, vec![3.0, 0.0, 0.0, 0.0]);

    println!();
    println!("All results verified.");
}
