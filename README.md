# dANN: Dendritic Artificial Neural Networks

This project explores the capabilities of dendritic artificial neural networks (dANNs), a biologically-inspired neural network architecture. The key idea is that individual neurons can perform complex non-linear computations, thanks to their dendritic structures. This repository provides a Go implementation of dANNs and demonstrates their effectiveness on various classification tasks.

## Getting Started

To run the experiments, you need to have Go installed on your system. You can download it from the official website: [https://golang.org/](https://golang.org/)

Once Go is installed, you can run any of the experiments by executing the corresponding `.go` file. For example, to run the XOR experiment, you would use the following command:

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_xor.go
```

## Experiments

This repository includes several experiments that showcase the dANN's ability to solve linearly non-separable problems. Each experiment is self-contained in its own `.go` file.

### 1. XOR Problem

This is the classic "hello world" of neural networks. The XOR problem is a simple binary classification task where the output is 1 if the inputs are different, and 0 otherwise.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_xor.go
```

**Expected Output:**

```
--- Training Neuron on XOR problem ---
Epoch 0, Error: 0.256691
Epoch 1000, Error: 0.142471
Epoch 2000, Error: 0.071312
Epoch 3000, Error: 0.005175
Epoch 4000, Error: 0.002470
Epoch 5000, Error: 0.001582
Epoch 6000, Error: 0.001151
Epoch 7000, Error: 0.000899
Epoch 8000, Error: 0.000734
Epoch 9000, Error: 0.000619

--- Training Complete ---

--- Testing Trained Neuron ---
Input: [0, 0], Expected: 0, Prediction: 0.0156, Rounded: 0
Input: [0, 1], Expected: 1, Prediction: 0.9787, Rounded: 1
Input: [1, 0], Expected: 1, Prediction: 0.9743, Rounded: 1
Input: [1, 1], Expected: 0, Prediction: 0.0279, Rounded: 0
```

### 2. Reimplemented XOR Problem with dANN

This experiment uses a full dANN to solve the XOR problem.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dann_reimplemented_xor.go
```

**Expected Output:**

```
--- Generating XOR Training Data for dANN ---
Generated 200 data points for XOR problem.

--- Training dANN on XOR problem ---
Epoch 5000, Avg Error: 0.000002
Epoch 10000, Avg Error: 0.000001
Epoch 15000, Avg Error: 0.000001
Epoch 20000, Avg Error: 0.000001
Epoch 25000, Avg Error: 0.000000
Epoch 29999, Avg Error: 0.000000

--- Training Complete ---

--- Testing Trained dANN ---
Input: [1, 1], Expected: 0, Prediction: 0.0006, Rounded: 0, Correct: true
...
Final Accuracy on XOR: 100.00%
```

### 3. Circle Problem

This experiment trains a dANN to classify whether a point is inside or outside a circle.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_circle.go
```

**Expected Output:**

```
--- Generating Circle Training Data ---
Generated 200 data points for a circle with radius 1.0

--- Training Neuron on Circle problem ---
Epoch 0, Error: 0.210609
...
Epoch 18000, Error: 0.000058

--- Training Complete ---

--- Testing Trained Neuron ---

Final Accuracy: 100.00%
...
```

### 4. Ring Problem

This experiment trains a dANN to classify whether a point is inside a ring.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_ring.go
```

**Expected Output:**

```
--- Generating Ring Training Data ---
Generated 500 data points for a ring (inner 0.5, outer 1.0).

--- Training Neuron on Ring problem ---
Epoch 5000, Error: 0.009130
...
Epoch 45000, Error: 0.002465

--- Training Complete ---

--- Testing Trained Neuron on Ring ---

Final Accuracy on Ring: 99.80%
```

### 5. Two Spirals Problem

This is a very challenging benchmark for classification algorithms. The dANN is trained to distinguish between two intertwined spirals.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_spirals.go
```

**Expected Output:**

```
--- Generating Two Spirals Training Data ---
Generated 200 data points for the two spirals problem.

--- Training Neuron on Two Spirals problem ---
Epoch 5000, Error: 0.006104
...
Epoch 45000, Error: 0.003999

--- Training Complete ---

--- Testing Trained Neuron on Spirals ---

Final Accuracy on Two Spirals: 99.00%
```

### 6. Two Circles Problem

This experiment trains a dANN to classify whether a point is inside one of two disjoint circles.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_two_circles.go
```

**Expected Output:**

```
--- Generating Two Disjoint Circles Training Data ---
Generated 500 data points for two circles (radius 0.8, centers -1.5,0.0 and 1.5,0.0).

--- Training Neuron on Two Disjoint Circles problem ---
Epoch 5000, Error: 0.000247
...
Epoch 45000, Error: 0.000018

--- Training Complete ---

--- Testing Trained Neuron on Two Disjoint Circles ---

Final Accuracy on Two Disjoint Circles: 100.00%
```

### 7. XOR Circles Problem

This experiment is a variation of the two circles problem, where the dANN has to learn the XOR of the two circles.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_xor_circles.go
```

**Expected Output:**

```
--- Generating XOR of Two Circles Training Data ---
Generated 500 data points for XOR of two circles (radius 1.0, centers -1.0,0.0 and 1.0,0.0).

--- Training Neuron on XOR of Two Circles problem ---
Epoch 5000, Error: 0.002814
...
Epoch 45000, Error: 0.002046

--- Training Complete ---

--- Testing Trained Neuron on XOR of Two Circles ---

Final Accuracy on XOR of Two Circles: 99.80%
```

### 8. Checkerboard Problem

This experiment trains a dANN to classify points on a checkerboard pattern.

**To run:**

```bash
go run C:/run/TestGemCli/dANN/dendritic_neuron_checkerboard.go
```

**Expected Output:**

```
--- Generating Checkerboard Training Data ---
Generated 500 data points for the checkerboard problem.

--- Training Neuron on Checkerboard problem ---
Epoch 5000, Error: 0.062449
...
Epoch 45000, Error: 0.059157

--- Training Complete ---

--- Testing Trained Neuron on Checkerboard ---

Final Accuracy on Checkerboard: 94.20%
```

## How it Works

The core of the implementation is the `DendriticNeuron` struct, which contains a set of `DendriticCompartment`s. Each compartment takes all the inputs and applies a non-linear function (tanh). The outputs of the compartments are then weighted and summed up in the soma, which applies a final non-linear function (sigmoid).

The training process uses backpropagation to adjust the weights of both the soma and the compartments.

## Further Reading

The PDF documents in this repository provide more background on dendritic neurons and their relationship to other neural network architectures.

*   **LunchnLearn-Dendritic Neuron Brief.pdf**: A brief introduction to dendritic neurons.
*   **Transformer Arch and Dendritic Networks.pdf**: A more in-depth paper that explores the relationship between transformer architectures and dendritic networks.
