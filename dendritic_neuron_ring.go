

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

// --- 4. New Data Generation for Ring Problem ---
func generateRingData(numPoints int, innerRadius, outerRadius float64) []TrainingData {
	data := make([]TrainingData, numPoints)
	for i := 0; i < numPoints; i++ {
		// Generate points within a range, e.g., [-outerRadius*1.2, outerRadius*1.2]
		x := (rand.Float64()*2 - 1) * outerRadius * 1.2
		y := (rand.Float64()*2 - 1) * outerRadius * 1.2

		inputs := []float64{x, y}

		// Check if the point is inside the ring
		distance := math.Sqrt(x*x + y*y)

		expected := 0.0
		if distance >= innerRadius && distance <= outerRadius {
			expected = 1.0
		}
		data[i] = TrainingData{inputs: inputs, expected: expected}
	}
	return data
}

// --- 5. Main Execution (Updated for Ring Problem) ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Generating Ring Training Data ---")
	innerR := 0.5
	outerR := 1.0
	ringData := generateRingData(500, innerR, outerR)
	fmt.Printf("Generated %d data points for a ring (inner %.1f, outer %.1f).\n\n",
		len(ringData), innerR, outerR)

	// Create a highly complex neuron with 16 compartments
	neuron := NewDendriticNeuron(2, 16)

	fmt.Println("--- Training Neuron on Ring problem ---")
	neuron.Train(ringData, 50000, 0.05)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained Neuron on Ring ---")
	correct := 0
	for _, d := range ringData {
		prediction, _ := neuron.Forward(d.inputs)
		roundedPrediction := math.Round(prediction)
		if roundedPrediction == d.expected {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(ringData)) * 100
	fmt.Printf("\nFinal Accuracy on Ring: %.2f%%\n", accuracy)
	time.Sleep(time.Second) // Add a small delay to ensure output flushes
}
