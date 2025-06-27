package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- 1. Dendritic Compartment ---
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

// --- 2. Dendritic Neuron ---
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	somaWeights  []float64
	somaBias     float64
}

func NewDendriticNeuron(numInputs, numCompartments int) *DendriticNeuron {
	comps := make([]*DendriticCompartment, numCompartments)
	for i := range comps {
		comps[i] = NewDendriticCompartment(numInputs)
	}
	sw := make([]float64, numCompartments)
	for i := range sw {
		sw[i] = rand.Float64()*2 - 1
	}
	return &DendriticNeuron{
		compartments: comps,
		somaWeights:  sw,
		somaBias:     rand.Float64()*2 - 1,
	}
}

func (dn *DendriticNeuron) Forward(inputs []float64) (float64, []float64) {
	outs := make([]float64, len(dn.compartments))
	for i, comp := range dn.compartments {
		outs[i] = comp.Process(inputs)
	}
	sum := dn.somaBias
	for i, o := range outs {
		sum += o * dn.somaWeights[i]
	}
	return 1.0 / (1.0 + math.Exp(-sum)), outs
}

// --- 3. Training Data ---
type TrainingData struct {
	inputs   []float64
	expected float64
}

// --- 4. Train with adaptive LR & early stopping ---
func (dn *DendriticNeuron) Train(data []TrainingData, epochs int, initLR float64, patience int, minDelta float64) {
	lr := initLR
	bestErr := math.MaxFloat64
	noImprovement := 0

	for e := 0; e < epochs; e++ {
		errSum := 0.0

		for _, d := range data {
			pred, outs := dn.Forward(d.inputs)
			err := d.expected - pred
			errSum += err * err

			// Backpropagate at soma
			delta := err * pred * (1.0 - pred)
			for i, o := range outs {
				dn.somaWeights[i] += lr * delta * o
			}
			dn.somaBias += lr * delta

			// Backpropagate to compartments
			for i, comp := range dn.compartments {
				ec := delta * dn.somaWeights[i]
				dc := ec * (1.0 - outs[i]*outs[i])
				for j, inp := range d.inputs {
					comp.weights[j] += lr * dc * inp
				}
				comp.bias += lr * dc
			}
		}

		avgErr := errSum / float64(len(data))

		// Check improvement
		if bestErr-avgErr > minDelta {
			bestErr = avgErr
			noImprovement = 0
		} else {
			noImprovement++
		}

		// Log progress every 1000 epochs
		if e%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %.6f, LR: %.5f\n", e, avgErr, lr)
		}

		// Early stopping
		if noImprovement >= patience {
			fmt.Printf("Early stopping at epoch %d, Error: %.6f\n", e, avgErr)
			return
		}

		// Decay learning rate every 2000 epochs
		if e > 0 && e%2000 == 0 {
			lr *= 0.9
		}
	}
}

// --- 5. Data Generation ---
func generateCircleData(n int, radius float64) []TrainingData {
	data := make([]TrainingData, n)
	for i := range data {
		x := rand.Float64()*4 - 2
		y := rand.Float64()*4 - 2
		label := 0.0
		if x*x+y*y < radius*radius {
			label = 1.0
		}
		data[i] = TrainingData{
			inputs:   []float64{x, y},
			expected: label,
		}
	}
	return data
}

// --- 6. Main ---
func main() {
	rand.Seed(time.Now().UnixNano())

	// Generate training data
	data := generateCircleData(200, 1.0)

	// Create neuron with 8 compartments
	neuron := NewDendriticNeuron(2, 8)

	// Train with max 20k epochs, init LR=0.1, patience=3000, minDelta=1e-4
	neuron.Train(data, 20000, 0.1, 3000, 1e-4)

	// Evaluate accuracy
	correct := 0
	for _, d := range data {
		pred, _ := neuron.Forward(d.inputs)
		if math.Round(pred) == d.expected {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(data)) * 100.0
	fmt.Printf("Final Accuracy: %.2f%%\n", accuracy)
}
