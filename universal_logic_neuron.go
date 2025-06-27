package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// --- Dendritic Compartment ---
type DendriticCompartment struct {
	weights []float64
	bias    float64
}

func NewDendriticCompartment(numInputs int) *DendriticCompartment {
	w := make([]float64, numInputs)
	for i := range w {
		w[i] = rand.Float64()*2 - 1
	}
	return &DendriticCompartment{weights: w, bias: rand.Float64()*2 - 1}
}

func (dc *DendriticCompartment) Process(inputs []float64) float64 {
	sum := dc.bias
	for i, x := range inputs {
		sum += x * dc.weights[i]
	}
	return math.Tanh(sum)
}

// --- Dendritic Neuron ---
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	somaWeights  []float64
	somaBias     float64

	labelWeights map[string]float64

	mu sync.Mutex
}

func NewDendriticNeuron(numInputs, numCompartments int) *DendriticNeuron {
	comps := make([]*DendriticCompartment, numCompartments)
	for i := range comps {
		comps[i] = NewDendriticCompartment(numInputs)
	}
	somaW := make([]float64, numCompartments)
	for i := range somaW {
		somaW[i] = rand.Float64()*2 - 1
	}
	return &DendriticNeuron{
		compartments: comps,
		somaWeights:  somaW,
		somaBias:     rand.Float64()*2 - 1,
		labelWeights: make(map[string]float64),
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (dn *DendriticNeuron) Forward(inputs []float64, label string) (float64, []float64) {
	compOuts := make([]float64, len(dn.compartments))
	for i, c := range dn.compartments {
		compOuts[i] = c.Process(inputs)
	}
	sum := dn.somaBias
	for i, o := range compOuts {
		sum += o * dn.somaWeights[i]
	}
	dn.mu.Lock()
	sum += dn.labelWeights[label] // Add label influence (lock for safe read)
	dn.mu.Unlock()
	return sigmoid(sum), compOuts
}

type TrainingData struct {
	inputs   []float64
	expected float64
	label    string
}

// --- Multithreaded training worker ---
func trainWorker(dn *DendriticNeuron, samples []TrainingData, lr float64, wg *sync.WaitGroup, loss *float64, lossMu *sync.Mutex) {
	defer wg.Done()

	localLoss := 0.0

	for _, d := range samples {
		pred, outs := dn.Forward(d.inputs, d.label)
		err := d.expected - pred
		localLoss += err * err
		delta := err * pred * (1 - pred)

		dn.mu.Lock()
		// Update soma weights & bias
		for i, o := range outs {
			dn.somaWeights[i] += lr * delta * o
		}
		dn.somaBias += lr * delta
		// Update label weights
		dn.labelWeights[d.label] += lr * delta
		// Update dendritic compartments
		for i, c := range dn.compartments {
			errC := delta * dn.somaWeights[i]
			deltaC := errC * (1 - outs[i]*outs[i])
			for j, x := range d.inputs {
				c.weights[j] += lr * deltaC * x
			}
			c.bias += lr * deltaC
		}
		dn.mu.Unlock()
	}

	lossMu.Lock()
	*loss += localLoss
	lossMu.Unlock()
}

// --- Training with multithreading, early stopping, adaptive LR ---
func (dn *DendriticNeuron) Train(data []TrainingData, epochs int, lr float64, patience int, minDelta float64) {
	bestError := math.MaxFloat64
	noImprovement := 0

	numCPU := runtime.NumCPU()

	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup
		totalLoss := 0.0
		var lossMu sync.Mutex

		chunkSize := len(data) / numCPU

		for i := 0; i < numCPU; i++ {
			start := i * chunkSize
			end := start + chunkSize
			if i == numCPU-1 {
				end = len(data)
			}
			wg.Add(1)
			go trainWorker(dn, data[start:end], lr, &wg, &totalLoss, &lossMu)
		}

		wg.Wait()

		avgError := totalLoss / float64(len(data))

		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %.6f, LR: %.5f\n", epoch, avgError, lr)
		}

		if bestError-avgError > minDelta {
			bestError = avgError
			noImprovement = 0
		} else {
			noImprovement++
		}

		if noImprovement >= patience {
			fmt.Printf("Early stop at epoch %d, Error: %.6f\n", epoch, avgError)
			break
		}

		if epoch > 0 && epoch%2000 == 0 {
			lr *= 0.9
		}
	}
}

// --- Logic Gate & Logic Block Data ---
func makeLogicGateData() []TrainingData {
	data := []TrainingData{}
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for _, in := range inputs {
		a, b := in[0], in[1]
		// Basic Gates
		data = append(data, TrainingData{in, a * b, "AND"})
		data = append(data, TrainingData{in, math.Max(a, b), "OR"})
		data = append(data, TrainingData{in, 1 - a, "NOT_A"})
		data = append(data, TrainingData{in, 1 - b, "NOT_B"})
		data = append(data, TrainingData{in, 1 - a*b, "NAND"})
		data = append(data, TrainingData{in, 1 - math.Max(a, b), "NOR"})
		data = append(data, TrainingData{in, float64(int(a) ^ int(b)), "XOR"})
		data = append(data, TrainingData{in, 1 - float64(int(a)^int(b)), "XNOR"})
		data = append(data, TrainingData{in, a, "BUFFER_A"})
		data = append(data, TrainingData{in, b, "BUFFER_B"})
		// Logic Blocks - Half Adder
		data = append(data, TrainingData{in, float64(int(a) ^ int(b)), "HALF_ADDER_SUM"})
		data = append(data, TrainingData{in, a * b, "HALF_ADDER_CARRY"})
	}
	return data
}

func makeFullAdderData() []TrainingData {
	data := []TrainingData{}
	for _, in := range [][]float64{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
		{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
	} {
		a, b, cin := in[0], in[1], in[2]
		sum := float64(int(a) ^ int(b) ^ int(cin))
		cout := float64((int(a)&int(b)) | (int(b)&int(cin)) | (int(a)&int(cin)))
		data = append(data, TrainingData{in, sum, "FULL_ADDER_SUM"})
		data = append(data, TrainingData{in, cout, "FULL_ADDER_CARRY"})
	}
	return data
}

// --- Main ---
func main() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())

	fmt.Println("Generating training data for gates and adders...")
	gateData := makeLogicGateData()
	adderData := makeFullAdderData()
	allData := append(gateData, adderData...)

	neuron := NewDendriticNeuron(3, 8) // use 3 inputs to handle full adder

	fmt.Println("Training universal neuron on all logic gates and blocks...")
	neuron.Train(allData, 20000, 0.1, 3000, 1e-4)

	fmt.Println("\nTesting trained neuron...\n")
	total, correct := 0, 0
	for _, d := range allData {
		pred, _ := neuron.Forward(d.inputs, d.label)
		predBinary := math.Round(pred)
		ok := predBinary == d.expected
		if ok {
			correct++
		}
		total++
		fmt.Printf("Label: %-18s Inputs: %v, Exp: %.0f, Pred: %.4f, R: %.0f, Correct: %v\n",
			d.label, d.inputs, d.expected, pred, predBinary, ok)
	}
	acc := float64(correct) / float64(total) * 100
	fmt.Printf("\nOverall Accuracy: %.2f%%\n", acc)
}
