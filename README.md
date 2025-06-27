Dendritic Neuron Model in Go

This repository contains implementations of dendritic neuron models in Go, progressively evolving from simple binary logic gates training to a universal neuron capable of learning multiple logic gates and combinational logic blocks simultaneously.
Features

    Dendritic Compartment & Neuron: Models with weighted inputs, nonlinear processing (tanh), and soma combining multiple compartments.

    Training with Backpropagation: Includes adaptive learning rate, early stopping, and error logging.

    Examples:

        XOR gate learning.

        Synthetic nonlinear data classification (circle data).

        Multi-label training on classical logic gates and combinational blocks (AND, OR, NOT, NAND, NOR, XOR, XNOR, BUFFER, Half Adder, Full Adder).

    Multi-threaded Training: Efficient parallel training using Go routines and synchronization.

    Accuracy Reporting: Evaluates and prints per-label accuracy and overall performance.

Usage

Run the main Go files to train the neuron on desired datasets and observe training progress and accuracy results. The multi-label universal neuron example demonstrates learning complex logic behaviors with one unified model.