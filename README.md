# Dendritic Neuron Classifier (Go)

This repository implements a biologically inspired **dendritic neuron** model in Go, capable of solving various logic gate and combinational logic block classification tasks.

## Features

- **Universal neuron** with multiple dendritic compartments and soma integration  
- Trains on all basic logic gates: AND, OR, NOT (on both inputs), NAND, NOR, XOR, XNOR, BUFFER  
- Supports common combinational logic blocks:
  - Half Adder (Sum and Carry)
  - Full Adder (Sum and Carry)  
- Multithreaded training leveraging all CPU cores for faster convergence  
- Adaptive learning rate with 10% decay every 2000 epochs  
- Detailed testing output showing input, expected output, predicted output, and correctness  
- Overall accuracy calculation after training  

## Prerequisites

- Go 1.18 or newer  
- Git (optional)

## Usage

Run the training and testing directly:

```bash
go run universal_logic_neuron.go
