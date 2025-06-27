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
	noImp := 0

	for e := 0; e < epochs; e++ {
		errSum := 0.0
		for _, d := range data {
			pred, outs := dn.Forward(d.inputs)
			err := d.expected - pred
			errSum += err * err

			// Soma update
			delta := err * pred * (1.0 - pred)
			for i, o := range outs {
				dn.somaWeights[i] += lr * delta * o
			}
			dn.somaBias += lr * delta

			// Compartment update
			for i, comp := range dn.compartments {
				ec := delta * dn.somaWeights[i]
				dc := ec * (1.0 - outs[i]*outs[i])
				for j, inp := range d.inputs {
					comp.weights[j] += lr * dc * inp
				}
				comp.bias += lr * dc
			}
		}

		avg := errSum / float64(len(data))

		// Check improvement
		if bestErr-avg > minDelta {
			bestErr = avg
			noImp = 0
		} else {
			noImp++
		}

		// Log every 1000 epochs
		if e%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %.6f, LR: %.5f\n", e, avg, lr)
		}

		// Early stopping
		if noImp >= patience {
			fmt.Printf("Early stopping at epoch %d, Error: %.6f\n", e, avg)
			break
		}

		// Decay LR every 2000 epochs
		if e > 0 && e%2000 == 0 {
			lr *= 0.9
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// XOR dataset
	data := []TrainingData{
		{inputs: []float64{0, 0}, expected: 0},
		{inputs: []float64{0, 1}, expected: 1},
		{inputs: []float64{1, 0}, expected: 1},
		{inputs: []float64{1, 1}, expected: 0},
	}

	neuron := NewDendriticNeuron(2, 4)
	neuron.Train(data, 10000, 0.1, 3000, 1e-4)

	fmt.Println("--- Predictions ---")
	for _, d := range data {
		pred, _ := neuron.Forward(d.inputs)
		fmt.Printf("Input: %v, Exp: %.0f, Pred: %.4f, R: %.0f\n",
			d.inputs, d.expected, pred, math.Round(pred))
	}
}
