

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- 1. Dendritic Compartment (Unchanged) ---
type DendriticCompartment struct {
	weights []float64
	bias    float64
}

func NewDendriticCompartment(numInputs int) *DendriticCompartment {
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1
	}
	return &DendriticCompartment{
		weights: weights,
		bias:    rand.Float64()*2 - 1,
	}
}

func (dc *DendriticCompartment) Process(inputs []float64) float64 {
	sum := dc.bias
	for i, input := range inputs {
		sum += input * dc.weights[i]
	}
	return math.Tanh(sum)
}

// --- 2. The Dendritic Neuron (Unchanged) ---
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	somaWeights  []float64
	somaBias     float64
}

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

func (dn *DendriticNeuron) Forward(inputs []float64) (float64, []float64) {
	compartmentOutputs := make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		compartmentOutputs[i] = compartment.Process(inputs)
	}

	finalSum := dn.somaBias
	for i, output := range compartmentOutputs {
		finalSum += output * dn.somaWeights[i]
	}

	finalOutput := 1.0 / (1.0 + math.Exp(-finalSum))
	return finalOutput, compartmentOutputs
}

// --- 3. Training (Unchanged) ---
type TrainingData struct {
	inputs   []float64
	expected float64
}

func (dn *DendriticNeuron) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for _, d := range data {
			prediction, compartmentOutputs := dn.Forward(d.inputs)
			err := d.expected - prediction
			totalError += err * err
			deltaSoma := err * prediction * (1 - prediction)
			for i, output := range compartmentOutputs {
				dn.somaWeights[i] += learningRate * deltaSoma * output
			}
			dn.somaBias += learningRate * deltaSoma
			for i, comp := range dn.compartments {
				errorCompartment := deltaSoma * dn.somaWeights[i]
				deltaCompartment := errorCompartment * (1 - math.Pow(compartmentOutputs[i], 2))
				for j, input := range d.inputs {
					comp.weights[j] += learningRate * deltaCompartment * input
				}
				comp.bias += learningRate * deltaCompartment
			}
		}
		if epoch > 0 && epoch%5000 == 0 {
			fmt.Printf("Epoch %d, Error: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- 4. New Data Generation for XOR of Two Circles ---
func generateXORCirclesData(numPoints int, radius float64, center1, center2 []float64) []TrainingData {
	data := make([]TrainingData, numPoints)
	for i := 0; i < numPoints; i++ {
		// Generate points within a range, e.g., [-3, 3]
		x := (rand.Float64()*2 - 1) * 3.0
		y := (rand.Float64()*2 - 1) * 3.0

		inputs := []float64{x, y}

		// Check if the point is inside each circle
		isInside1 := math.Pow(x-center1[0], 2)+math.Pow(y-center1[1], 2) < radius*radius
		isInside2 := math.Pow(x-center2[0], 2)+math.Pow(y-center2[1], 2) < radius*radius

		// Apply XOR logic
		expected := 0.0
		if (isInside1 && !isInside2) || (!isInside1 && isInside2) {
			expected = 1.0
		}
		data[i] = TrainingData{inputs: inputs, expected: expected}
	}
	return data
}

// --- 5. Main Execution (Updated for XOR of Two Circles) ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Generating XOR of Two Circles Training Data ---")
	radius := 1.0
	center1 := []float64{-1.0, 0.0}
	center2 := []float64{1.0, 0.0}
	xorCirclesData := generateXORCirclesData(500, radius, center1, center2)
	fmt.Printf("Generated %d data points for XOR of two circles (radius %.1f, centers %.1f,%.1f and %.1f,%.1f).\n\n",
		len(xorCirclesData), radius, center1[0], center1[1], center2[0], center2[1])

	// Create a highly complex neuron with 16 compartments
	neuron := NewDendriticNeuron(2, 16)

	fmt.Println("--- Training Neuron on XOR of Two Circles problem ---")
	neuron.Train(xorCirclesData, 50000, 0.05)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained Neuron on XOR of Two Circles ---")
	correct := 0
	for _, d := range xorCirclesData {
		prediction, _ := neuron.Forward(d.inputs)
		roundedPrediction := math.Round(prediction)
		if roundedPrediction == d.expected {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(xorCirclesData)) * 100
	fmt.Printf("\nFinal Accuracy on XOR of Two Circles: %.2f%%\n", accuracy)
	time.Sleep(time.Second) // Add a small delay to ensure output flushes
}
