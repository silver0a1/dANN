

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

// --- 4. New Data Generation for Spirals ---
func generateSpiralData(points int, noise float64) []TrainingData {
	data := make([]TrainingData, points*2)
	for i := 0; i < points; i++ {
		// Spiral 1 (Class 0)
		angle1 := float64(i) / float64(points) * 5.0
		radius1 := angle1
		x1 := radius1 * math.Cos(angle1+math.Pi) + rand.Float64()*noise
		y1 := radius1 * math.Sin(angle1+math.Pi) + rand.Float64()*noise
		data[i] = TrainingData{inputs: []float64{x1, y1}, expected: 0}

		// Spiral 2 (Class 1)
		angle2 := float64(i) / float64(points) * 5.0
		radius2 := angle2
		x2 := radius2 * math.Cos(angle2) + rand.Float64()*noise
		y2 := radius2 * math.Sin(angle2) + rand.Float64()*noise
		data[i+points] = TrainingData{inputs: []float64{x2, y2}, expected: 1}
	}
	return data
}

// --- 5. Main Execution (Updated for Spirals) ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Generating Two Spirals Training Data ---")
	spiralData := generateSpiralData(100, 0.2)
	fmt.Printf("Generated %d data points for the two spirals problem.\n\n", len(spiralData))

	// Create a highly complex neuron with 16 compartments
	neuron := NewDendriticNeuron(2, 16)

	fmt.Println("--- Training Neuron on Two Spirals problem ---")
	neuron.Train(spiralData, 50000, 0.05)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained Neuron on Spirals ---")
	correct := 0
	for _, d := range spiralData {
		prediction, _ := neuron.Forward(d.inputs)
		roundedPrediction := math.Round(prediction)
		if roundedPrediction == d.expected {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(spiralData)) * 100
	fmt.Printf("\nFinal Accuracy on Two Spirals: %.2f%%\n", accuracy)
	time.Sleep(time.Second) // Add a small delay to ensure output flushes
}
