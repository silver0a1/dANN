package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ActivationFunction defines the interface for activation functions.
type ActivationFunction interface {
	Apply(float64) float64
	Derivative(float64) float64
}

// LeakyReLU is the Leaky Rectified Linear Unit activation function.
type LeakyReLU struct {
	alpha float64
}

// Apply applies the Leaky ReLU function.
func (lr *LeakyReLU) Apply(x float64) float64 {
	if x > 0 {
		return x
	}
	return lr.alpha * x
}

// Derivative calculates the derivative of the Leaky ReLU function.
func (lr *LeakyReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return lr.alpha
}

// Dendrite represents a single dendrite in the dANN.
type Dendrite struct {
	weights []float64
	bias    float64
	inputs  []int
}

// Soma represents a single soma in the dANN.
type Soma struct {
	dendrites []*Dendrite
}

// DANN represents the Dendritic Artificial Neural Network.
type DANN struct {
	somas       []*Soma
	outputLayer []*Dendrite // Using Dendrite struct for simplicity in output layer
	activation  ActivationFunction
}

// NewDANN creates a new dANN with the specified architecture.
func NewDANN(inputSize, numSomas, dendritesPerSoma, synapsesPerDendrite, outputSize int, activation ActivationFunction) *DANN {
	rand.Seed(time.Now().UnixNano())

	dann := &DANN{
		somas:       make([]*Soma, numSomas),
		outputLayer: make([]*Dendrite, outputSize),
		activation:  activation,
	}

	// Initialize somas and dendrites
	for i := 0; i < numSomas; i++ {
		soma := &Soma{
			dendrites: make([]*Dendrite, dendritesPerSoma),
		}
		for j := 0; j < dendritesPerSoma; j++ {
			dendrite := &Dendrite{
				weights: make([]float64, synapsesPerDendrite),
				bias:    rand.Float64()*2 - 1,
				inputs:  make([]int, synapsesPerDendrite),
			}
			// Random input sampling (dANN-R)
			for k := 0; k < synapsesPerDendrite; k++ {
				dendrite.inputs[k] = rand.Intn(inputSize)
				dendrite.weights[k] = rand.Float64()*2 - 1
			}
			soma.dendrites[j] = dendrite
		}
		dann.somas[i] = soma
	}

	// Initialize output layer
	for i := 0; i < outputSize; i++ {
		outputNode := &Dendrite{
			weights: make([]float64, numSomas),
			bias:    rand.Float64()*2 - 1,
			inputs:  make([]int, numSomas),
		}
		for j := 0; j < numSomas; j++ {
			outputNode.inputs[j] = j
			outputNode.weights[j] = rand.Float64()*2 - 1
		}
		dann.outputLayer[i] = outputNode
	}

	return dann
}

// Forward performs the forward pass through the network.
func (d *DANN) Forward(input []float64) []float64 {
	somaOutputs := make([]float64, len(d.somas))
	for i, soma := range d.somas {
		dendriteOutputs := make([]float64, len(soma.dendrites))
		for j, dendrite := range soma.dendrites {
			sum := dendrite.bias
			for k, weight := range dendrite.weights {
				sum += weight * input[dendrite.inputs[k]]
			}
			dendriteOutputs[j] = d.activation.Apply(sum)
		}

		somaSum := 0.0
		for _, out := range dendriteOutputs {
			somaSum += out
		}
		somaOutputs[i] = d.activation.Apply(somaSum)
	}

	finalOutputs := make([]float64, len(d.outputLayer))
	for i, outputNode := range d.outputLayer {
		sum := outputNode.bias
		for j, weight := range outputNode.weights {
			sum += weight * somaOutputs[outputNode.inputs[j]]
		}
		finalOutputs[i] = sum
	}

	// Softmax for output layer
	return softmax(finalOutputs)
}

// softmax applies the softmax function to a slice of floats.
func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make([]float64, len(x))
	sumExps := 0.0
	for i, v := range x {
		exps[i] = math.Exp(v - maxVal)
		sumExps += exps[i]
	}

	for i := range exps {
		exps[i] /= sumExps
	}
	return exps
}

func main() {
	// Example usage
	inputSize := 784         // For a 28x28 image
	numSomas := 128
	dendritesPerSoma := 16
	synapsesPerDendrite := 16
	outputSize := 10         // For 10 classes
	leakyReLU := &LeakyReLU{alpha: 0.01}

	dann := NewDANN(inputSize, numSomas, dendritesPerSoma, synapsesPerDendrite, outputSize, leakyReLU)

	// Create a dummy input
	dummyInput := make([]float64, inputSize)
	for i := range dummyInput {
		dummyInput[i] = rand.Float64()
	}

	output := dann.Forward(dummyInput)

	fmt.Println("dANN Output:", output)
}