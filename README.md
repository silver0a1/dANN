# **dANN: A Simple Deep Learning Framework in Go**

This document provides comprehensive documentation for the dANN project, a lightweight deep learning framework built from scratch in Go.

## **1\. Project Overview**

dANN is a minimalist neural network library designed for simplicity, transparency, and educational purposes. It allows users to build, train, and use feedforward neural networks (multi-layer perceptrons) in a clean, idiomatic Go environment.  
The project's primary goal is to demystify the core concepts of neural networks—such as layers, activation functions, feedforward propagation, and backpropagation—by providing a clear and concise implementation. It relies on the gonum/mat package for its underlying matrix operations, which is the standard for numerical computing in Go.

### **Key Features**

* **Written in Go:** A clean, statically-typed, and concurrent-ready implementation.  
* **Simple API:** A straightforward and intuitive interface for model creation, training, and prediction.  
* **Customizable Architecture:** Easily define network architecture by adding layers with specified neuron counts and activation functions.  
* **Standard Activation Functions:** Includes Sigmoid and ReLU.  
* **Common Cost Functions:** Supports Mean Squared Error (MSE) and Cross-Entropy.  
* **Educational:** The code is well-structured and serves as an excellent learning tool for understanding deep learning fundamentals in a Go context.

## **2\. How It Works: Core Concepts**

The framework is built around the Network struct and leverages interfaces for Activation and Cost functions.

* **Network Struct**: This is the main struct that encapsulates the entire neural network. It manages:  
  * A slice of \*mat.Dense for the **weights** of each layer.  
  * A slice of \*mat.Dense for the **biases** of each layer.  
  * A slice of Activation interfaces, defining the activation function for each layer.  
  * The Cost function interface used to calculate error.  
* **Activation Interface**: This interface defines the behavior for activation functions.  
  * Apply(mat.Matrix) mat.Matrix: Applies the function to the input matrix (the weighted sum).  
  * Derivative(mat.Matrix) mat.Matrix: Calculates the derivative of the function, which is crucial for backpropagation.  
  * *Implementations*: Sigmoid, ReLU.  
* **Cost Interface**: This interface defines the behavior for cost functions.  
  * Fn(predicted, actual mat.Matrix) float64: Calculates the cost (error) between the predicted and actual outputs.  
  * Derivative(predicted, actual mat.Matrix) mat.Matrix: Calculates the derivative of the cost function.  
  * *Implementations*: MSE (Mean Squared Error), CrossEntropy.

## **3\. Getting Started**

### **Installation**

First, ensure you have Go installed on your system. Then, you can fetch the package and its dependency (gonum/mat) using the go get command.  
\# Get the dANN library  
go get github.com/silver0a1/dANN

\# The gonum/mat dependency will be fetched automatically.

### **Quick Start: Building a Model**

Here is how to define, train, and use a simple network in Go to solve the classic XOR problem. This example is adapted from cmd/xor/main.go in the repository.  
package main

import (  
	"fmt"  
	"math/rand"  
	"time"

	"github.com/silver0a1/dANN"  
	"gonum.org/v1/gonum/mat"  
)

func main() {  
	// Seed the random number generator  
	rand.Seed(time.Now().UnixNano())

	// 1\. Define the training data  
	// Input data for XOR  
	data := \[\]\[\]\[\]float64{  
		{{0}, {0}}, {{0}, {1}}, {{1}, {0}}, {{1}, {1}},  
	}  
	// Expected output (labels)  
	labels := \[\]\[\]\[\]float64{  
		{{0}}, {{1}}, {{1}}, {{0}},  
	}

	// 2\. Create a dANN model instance  
	// We use Mean Squared Error (MSE) as the cost function  
	network := dann.New(dann.MSE)

	// 3\. Define the network architecture by adding layers  
	// The first \`Add\` implicitly defines the input layer size.  
	network.Add(2, dann.ReLU)  // Hidden layer with 2 neurons  
	network.Add(1, dann.Sigmoid) // Output layer with 1 neuron

	// 4\. Train the network  
	network.Train(data, labels, 1000, 1, 0.1)

	// 5\. Make predictions on new data  
	for \_, d := range data {  
		input := mat.NewDense(len(d), len(d\[0\]), nil)  
		for i, v := range d {  
			input.SetRow(i, v)  
		}  
		prediction := network.Predict(input)  
		fmt.Printf("Input: %v, Prediction: %v\\n", d, mat.Formatted(prediction, mat.Prefix("    ")))  
	}  
}

## **4\. API Reference**

### **dann.New(cost Cost) \*Network**

* **Description**: Creates and initializes a new Network instance.  
* **Parameters**:  
  * cost (Cost): The cost function to use (e.g., dann.MSE, dann.CrossEntropy).

### **(\*Network) Add(neurons int, activation Activation)**

* **Description**: Adds a new layer to the network. The number of neurons in the previous layer (or the input size for the first layer) is inferred automatically.  
* **Parameters**:  
  * neurons (int): The number of neurons in this new layer.  
  * activation (Activation): The activation function for this layer (e.g., dann.Sigmoid, dann.ReLU).

### **(\*Network) Train(data, labels \[\]\[\]\[\]float64, epochs, batchSize int, learningRate float64)**

* **Description**: Trains the model on the provided dataset.  
* **Parameters**:  
  * data (\[\]\[\]\[\]float64): The input training data.  
  * labels (\[\]\[\]\[\]float64): The target output values.  
  * epochs (int): The number of times to iterate over the entire dataset.  
  * batchSize (int): The number of samples to process before updating weights.  
  * learningRate (float64): The step size for updating weights.

### **(\*Network) Predict(data mat.Matrix) mat.Matrix**

* **Description**: Generates a prediction for a single input data point.  
* **Parameters**:  
  * data (mat.Matrix): The input data to predict on.  
* **Returns**: A mat.Matrix containing the model's output.
