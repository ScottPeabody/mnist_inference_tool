use ndarray::{Array, ArrayBase, ArrayViewD, Dim, OwnedRepr};
use ort::{GraphOptimizationLevel, Session};
use std::time::Instant;
use std::env;

fn main() -> ort::Result<()> {
    // Retrieve command-line arguments, keeping simple for now.
    let args: Vec<String> = env::args().collect();

    let start_time = Instant::now();

    // Load model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("./mnist-12.onnx")?;

    let setup_duration = start_time.elapsed();

    // Load and preprocess the image
    let image_path = &args[1];
    let preprocess_start = Instant::now();
    let preprocessed = preprocess_image(image_path);
    let preprocess_duration = preprocess_start.elapsed();

    let inputs = ort::inputs! {
        "Input3" => preprocessed
    }.unwrap();

    // Run inference with the prepared input tensor
    let infer_start = Instant::now();
    let outputs = session.run(inputs).unwrap();
    let infer_duration = infer_start.elapsed();

    // Postprocess inference output tensor
    let post_process_start = Instant::now();
    let mut probabilities: Option<Vec<f32>> = None;
    outputs.get("Plus214_Output_0").map_or_else(
        || eprintln!("Output tensor 'Plus214_Output_0' not found"),
        |output_tensor| {
            output_tensor.try_extract_tensor::<f32>().map_or_else(
                |err| eprintln!("Failed to extract tensor: {:?}", err),
                |logits: ArrayBase<ndarray::ViewRepr<&f32>, Dim<ndarray::IxDynImpl>>| {
                    println!("Logits: {:?}", logits);
                    probabilities = postprocess(logits.view());
                }
            )
        }
    );
    let post_process_duration = post_process_start.elapsed();

    if let Some(probabilities) = probabilities {
        log_probabilities(&probabilities);
    } else {
        eprintln!("Failed to process logits or no probabilities found.");
    }

    let total_duration = start_time.elapsed();
    println!("Session Setup time: {:.2?}", setup_duration);
    println!("Preprocessing time: {:.2?}", preprocess_duration);
    println!("Inference time: {:.2?}", infer_duration);
    println!("Postprocessing time: {:.2?}", post_process_duration);
    println!("Aggregated Duration Without Setup: {:.2?}", preprocess_duration + infer_duration + post_process_duration);
    println!("Total application runtime: {:.2?}", total_duration);

    Ok(())
}

fn preprocess_image(image_path: &str) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    // Load the image and resize directly to 28x28 grayscale
    let img = image::open(image_path).unwrap();
    let resized = img.resize_exact(28, 28, image::imageops::FilterType::Triangle).into_luma8();

    // Initialize a new ndarray and populate it with pixel values
    let tensor = Array::from_shape_fn((1, 1, 28, 28), |(_, _, y, x)| {
        let pixel_value = resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        pixel_value
    });

    tensor
}

fn postprocess(logits: ArrayViewD<f32>) -> Option<Vec<f32>> {
    match logits.as_slice() {
        Some(logits_slice) => Some(softmax(logits_slice)),
        None => {
            eprintln!("Failed to get a slice of the logits.");
            None
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum_exps = 0.0;

    for &logit in logits {
        let exp_val = (logit - max_logit).exp();
        exps.push(exp_val);
        sum_exps += exp_val;
    }

    exps.iter_mut().for_each(|x| *x /= sum_exps);
    exps
}

fn log_probabilities(probabilities: &Vec<f32>) {
    // Print probabilities for each digit
    for (i, &prob) in probabilities.iter().enumerate() {
        println!("Digit {}: Probability {:.6}", i, prob);
    }

    // Find the digit with the highest probability
    if let Some((predicted_digit, _)) = probabilities.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()) {
        println!("Predicted digit: {}", predicted_digit);
    } else {
        println!("Could not determine the predicted digit.");
    }
}