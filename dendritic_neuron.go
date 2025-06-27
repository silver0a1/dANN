

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- 1. Dendritic Compartment ---
// A small computational unit within the neuron.
type DendriticCompartment struct {
	weights []float64
	bias    float64
}

// NewDendriticCompartment creates a compartment with random weights and bias.
func NewDendriticCompartment(numInputs int) *DendriticCompartment {
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1 // Random weights between -1 and 1
	}
	return &DendriticCompartment{
		weights: weights,
		bias:    rand.Float64()*2 - 1,
	}
}

// Process applies a non-linear activation to the weighted sum of its inputs.
func (dc *DendriticCompartment) Process(inputs []float64) float64 {
	sum := dc.bias
	for i, input := range inputs {
		sum += input * dc.weights[i]
	}
	return math.Tanh(sum) // Tanh activation for compartments
}

// --- 2. The Dendritic Neuron ---
// Our more powerful neuron model.
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	somaWeights  []float64
	somaBias     float64
}

// NewDendriticNeuron creates a neuron with a specified number of compartments.
func NewDendriticNeuron(numInputs, numCompartments int) *DendriticNeuron {
	compartments := make([]*DendriticCompartment, numCompartments)
	for i := 0; i < numCompartments; i++ {
		compartments[i] = NewDendriticCompartment(numInputs)
	}

	somaWeights := make([]float64, numCompartments)
	for i := range somaWeights {
		somaWeights[i] = rand.Float64()*2 - 1
	}

	return &DendriticNeuron{
		compartments: compartments,
		somaWeights:  somaWeights,
		somaBias:     rand.Float64()*2 - 1,
	}
}

// Forward performs the two-stage computation.
func (dn *DendriticNeuron) Forward(inputs []float64) (float64, []float64) {
	compartmentOutputs := make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		compartmentOutputs[i] = compartment.Process(inputs)
	}

	finalSum := dn.somaBias
	for i, output := range compartmentOutputs {
		finalSum += output * dn.somaWeights[i]
	}

	// Sigmoid activation for the final output
	finalOutput := 1.0 / (1.0 + math.Exp(-finalSum))
	return finalOutput, compartmentOutputs
}

// --- 3. Training ---
type TrainingData struct {
	inputs   []float64
	expected float64
}

// Train adjusts the neuron's weights and biases based on the provided data.
func (dn *DendriticNeuron) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for _, d := range data {
			// Forward pass
			prediction, compartmentOutputs := dn.Forward(d.inputs)

			// Calculate error
			err := d.expected - prediction
			totalError += err * err

			// --- Backward pass (Learning) ---

			// 1. Update Soma weights and bias
			// The derivative of the sigmoid function is output * (1 - output)
			deltaSoma := err * prediction * (1 - prediction)
			for i, output := range compartmentOutputs {
				dn.somaWeights[i] += learningRate * deltaSoma * output
			}
			dn.somaBias += learningRate * deltaSoma

			// 2. Update Compartment weights and bias
			// The derivative of tanh is 1 - output^2
			for i, comp := range dn.compartments {
				// Propagate the error back to the compartment
				errorCompartment := deltaSoma * dn.somaWeights[i]
				deltaCompartment := errorCompartment * (1 - math.Pow(compartmentOutputs[i], 2))
				for j, input := range d.inputs {
					comp.weights[j] += learningRate * deltaCompartment * input
				}
				comp.bias += learningRate * deltaCompartment
			}
		}
		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- 4. Main Execution ---
func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// XOR problem training data
	xorData := []TrainingData{
		{inputs: []float64{0, 0}, expected: 0},
		{inputs: []float64{0, 1}, expected: 1},
		{inputs: []float64{1, 0}, expected: 1},
		{inputs: []float64{1, 1}, expected: 0},
	}

	// Create a new neuron with 2 inputs and 4 compartments.
	// More compartments can sometimes help find a solution faster.
	neuron := NewDendriticNeuron(2, 4)

	fmt.Println("--- Training Neuron on XOR problem ---")
	neuron.Train(xorData, 10000, 0.1)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained Neuron ---")
	for _, d := range xorData {
		prediction, _ := neuron.Forward(d.inputs)
		fmt.Printf("Input: [%.0f, %.0f], Expected: %.0f, Prediction: %.4f, Rounded: %.0f\n",
			d.inputs[0], d.inputs[1], d.expected, prediction, math.Round(prediction))
	}
}
