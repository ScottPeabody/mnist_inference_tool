# MNIST Model Inference with Rust and ONNX

This project demonstrates how to perform inference on an MNIST model using Rust with the ONNX runtime (`ort` crate). The Rust code loads an ONNX model trained on the MNIST dataset and performs inference on a given image file.

## Requirements
- Rust programming environment

## General Details
- `ort` crate for ONNX runtime
    - https://github.com/pykeio/ort
    - https://github.com/onnx/models/tree/main/validated/vision/classification/mnist

#### The MNIST Model has the following properties:
```
Name: CNTKGraph
Description:
Produced by onnx.quantize
Inputs:
    0 Input3: Tensor<f32>(1, 1, 28, 28)
Outputs:
    0 Plus214_Output_0: Tensor<f32>(1, 10)
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mnist-rust-onnx.git
   cd mnist-rust-onnx
   ```

2. Build and run the project:

   ```bash
   cargo build --release
   ./target/release/mnist-rust-onnx path/to/your/image.png
   ```

   Replace `path/to/your/image.png` with the path to the image file you want to predict.

3. I also have a `mnist-onnxruntime.ipynb` juypter notebook in here which I used for sanity checking myself as I worked through the rust implementation. Interesting to note that the ors load in rust is really slow and in python I think might be included in the overhead when it starts up so it makes it tricky to compare performance at a glance.

## Included Files
   - I included some test_digit_data from http://yann.lecun.com/exdb/mnist/ for your convenience.
   - It is found in test_digit_data. `cargo run ./test_digit_data/8/6` where `/8` is the digit and `/6` is a varied instance.

### Example MNIST Digits

![MNIST Examples](MnistExamples.png)

## Implementation Details

### Preprocessing

The `preprocess_image` function loads and preprocesses the input image:
- Resizes the image to 28x28 pixels
- Converts it to grayscale and normalizes pixel values to the range [0, 1]

### Inference

- The ONNX model (`mnist-12.onnx`) is included for you for convenience and it loaded and configured with optimizations.
- Image data is fed into the model, and inference is performed.
- The output logits are extracted and processed using softmax to get probabilities for each digit (0-9).

### Postprocessing

- The `postprocess` function receives logits from the model's output.
- Applies softmax to obtain probabilities and determines the predicted digit based on the highest probability.

## Performance Metrics

The application prints various performance metrics and I am using to optimize the code.
- Setup time for loading the model
- Preprocessing time for image resizing and normalization
- Inference time for running the model
- Post-processing time for calculating probabilities and determining predictions
- Aggregated duration without setup time
- Total application runtime

## Example Output

```plaintext
Running `target\debug\mnist_inference_tool.exe ./test_digit_data/8/6.jpg`
Logits: [[0.41888416, -6.7712946, 8.670402, 1.053951, -4.0479784, -0.7836626, -1.1376514, -14.303154, 15.571114, -1.3362999]], shape=[1, 10], strides=[10, 1], layout=CFcf (0xf), dynamic ndim=2
Digit 0: Probability 0.000000
Digit 1: Probability 0.000000
Digit 2: Probability 0.001006
Digit 3: Probability 0.000000
Digit 4: Probability 0.000000
Digit 5: Probability 0.000000
Digit 6: Probability 0.000000
Digit 7: Probability 0.000000
Digit 8: Probability 0.998993
Digit 9: Probability 0.000000
Predicted digit: 8
Session Setup time: 15.89ms
Preprocessing time: 613.90µs
Inference time: 118.90µs
Post-processing time: 851.20µs
Aggregated Duration Without Setup: 1.58ms
Total application runtime: 17.50ms
```
```plaintext
Running `target\debug\mnist_inference_tool.exe ./test_digit_data/9/2.jpg`
Logits: [[-7.8424406, -12.053177, -12.620951, -3.9534318, 13.431284, -3.05846, -10.257339, 3.125442, 2.2841487, 24.8287]], shape=[1, 10], strides=[10, 1], layout=CFcf (0xf), dynamic ndim=2
Digit 0: Probability 0.000000
Digit 1: Probability 0.000000
Digit 2: Probability 0.000000
Digit 3: Probability 0.000000
Digit 4: Probability 0.000011
Digit 5: Probability 0.000000
Digit 6: Probability 0.000000
Digit 7: Probability 0.000000
Digit 8: Probability 0.000000
Digit 9: Probability 0.999989
Predicted digit: 9
Session Setup time: 16.52ms
Preprocessing time: 602.50µs
Inference time: 139.10µs
Post-processing time: 835.30µs
Aggregated Duration Without Setup: 1.58ms
Total application runtime: 18.12ms
```