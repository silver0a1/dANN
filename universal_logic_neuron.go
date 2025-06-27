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
		w[i] = rand.Float64()*0.1 - 0.05
	}
	return &DendriticCompartment{weights: w, bias: rand.Float64()*0.1 - 0.05}
}

func (dc *DendriticCompartment) Process(inputs []float64) float64 {
	sum := dc.bias
	for i, x := range inputs {
		sum += x * dc.weights[i]
	}
	return math.Tanh(sum)
}

// --- Dendritic Neuron with multi-output ---
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	// For each output label, there is a separate soma weight vector and bias
	somaWeights [][]float64 // [numLabels][numCompartments]
	somaBias    []float64   // [numLabels]

	mu sync.RWMutex
}

func NewDendriticNeuron(numInputs, numCompartments, numOutputs int) *DendriticNeuron {
	comps := make([]*DendriticCompartment, numCompartments)
	for i := range comps {
		comps[i] = NewDendriticCompartment(numInputs)
	}

	somaWeights := make([][]float64, numOutputs)
	somaBias := make([]float64, numOutputs)
	for i := 0; i < numOutputs; i++ {
		somaWeights[i] = make([]float64, numCompartments)
		for j := 0; j < numCompartments; j++ {
			somaWeights[i][j] = rand.Float64()*0.1 - 0.05
		}
		somaBias[i] = rand.Float64()*0.1 - 0.05
	}

	return &DendriticNeuron{
		compartments: comps,
		somaWeights:  somaWeights,
		somaBias:     somaBias,
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Forward returns slice of outputs for all labels, plus compartment outputs for backprop
func (dn *DendriticNeuron) Forward(inputs []float64) ([]float64, []float64) {
	compOuts := make([]float64, len(dn.compartments))
	for i := range dn.compartments {
		compOuts[i] = dn.compartments[i].Process(inputs)
	}

	outputs := make([]float64, len(dn.somaWeights))
	dn.mu.RLock()
	for i := range dn.somaWeights {
		sum := dn.somaBias[i]
		for j, o := range compOuts {
			sum += o * dn.somaWeights[i][j]
		}
		outputs[i] = sigmoid(sum)
	}
	dn.mu.RUnlock()

	return outputs, compOuts
}

// --- Training Data with multi-label expected outputs ---
type TrainingData struct {
	inputs   []float64
	expected []float64 // Expected outputs per label
}

// Labels and their indices for output vector
var labels = []string{
	"AND", "OR", "NOT_A", "NOT_B", "NAND", "NOR", "XOR", "XNOR",
	"BUFFER_A", "BUFFER_B", "HALF_ADDER_SUM", "HALF_ADDER_CARRY",
	"FULL_ADDER_SUM", "FULL_ADDER_CARRY",
}

// Map label name to index for training
var labelIndexMap map[string]int

func init() {
	labelIndexMap = make(map[string]int)
	for i, l := range labels {
		labelIndexMap[l] = i
	}
}

// --- Input normalization ---
func normalizeInput(x float64) float64 {
	// keep 0 or 1 as is
	return x
}

// --- Generate multi-label training data ---
func makeMultiLabelData() []TrainingData {
	rawData := []struct {
		inputs []float64
		label  string
		value  float64
	}{}

	// Logic gate inputs for 2-input gates
	inputs2 := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for _, in := range inputs2 {
		a, b := normalizeInput(in[0]), normalizeInput(in[1])
		rawData = append(rawData, []struct {
			inputs []float64
			label  string
			value  float64
		}{
			{in, "AND", a * b},
			{in, "OR", math.Max(a, b)},
			{in, "NOT_A", 1 - a},
			{in, "NOT_B", 1 - b},
			{in, "NAND", 1 - a*b},
			{in, "NOR", 1 - math.Max(a, b)},
			{in, "XOR", float64(int(a) ^ int(b))},
			{in, "XNOR", 1 - float64(int(a)^int(b))},
			{in, "BUFFER_A", a},
			{in, "BUFFER_B", b},
			{in, "HALF_ADDER_SUM", float64(int(a) ^ int(b))},
			{in, "HALF_ADDER_CARRY", a * b},
		}...)
	}

	// 3-input full adder data
	inputs3 := [][]float64{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
		{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
	}
	for _, in := range inputs3 {
		a, b, cin := normalizeInput(in[0]), normalizeInput(in[1]), normalizeInput(in[2])
		sum := float64(int(a) ^ int(b) ^ int(cin))
		cout := float64((int(a)&int(b)) | (int(b)&int(cin)) | (int(a)&int(cin)))
		rawData = append(rawData,
			struct {
				inputs []float64
				label  string
				value  float64
			}{in, "FULL_ADDER_SUM", sum},
		)
		rawData = append(rawData,
			struct {
				inputs []float64
				label  string
				value  float64
			}{in, "FULL_ADDER_CARRY", cout},
		)
	}

	// Now aggregate rawData by inputs to create multi-label output vectors
	grouped := make(map[string]*TrainingData)
	for _, d := range rawData {
		key := fmt.Sprintf("%v", d.inputs)
		td, exists := grouped[key]
		if !exists {
			td = &TrainingData{
				inputs:   d.inputs,
				expected: make([]float64, len(labels)),
			}
			grouped[key] = td
		}
		idx := labelIndexMap[d.label]
		td.expected[idx] = d.value
	}

	result := make([]TrainingData, 0, len(grouped))
	for _, td := range grouped {
		result = append(result, *td)
	}
	return result
}

// --- Training worker for multi-output neuron ---
func trainWorker(dn *DendriticNeuron, samples []TrainingData, lr float64, wg *sync.WaitGroup, loss *float64, lossMu *sync.Mutex) {
	defer wg.Done()

	localLoss := 0.0

	// Gradients per output label
	somaWeightGrads := make([][]float64, len(dn.somaWeights))
	somaBiasGrads := make([]float64, len(dn.somaBias))
	for i := range somaWeightGrads {
		somaWeightGrads[i] = make([]float64, len(dn.somaWeights[i]))
	}
	compWeightGrads := make([][]float64, len(dn.compartments))
	compBiasGrads := make([]float64, len(dn.compartments))
	for i := range dn.compartments {
		compWeightGrads[i] = make([]float64, len(dn.compartments[i].weights))
	}

	for _, d := range samples {
		outputs, compOuts := dn.Forward(d.inputs)

		for labelIdx := range outputs {
			err := d.expected[labelIdx] - outputs[labelIdx]
			localLoss += err * err
			delta := err * outputs[labelIdx] * (1 - outputs[labelIdx])

			// Update soma gradients
			for i, compOut := range compOuts {
				somaWeightGrads[labelIdx][i] += delta * compOut
			}
			somaBiasGrads[labelIdx] += delta

			// Update dendritic compartment gradients
			for i := range dn.compartments {
				errC := delta * dn.somaWeights[labelIdx][i]
				deltaC := errC * (1 - compOuts[i]*compOuts[i])
				for j, x := range d.inputs {
					compWeightGrads[i][j] += deltaC * x
				}
				compBiasGrads[i] += deltaC
			}
		}
	}

	dn.mu.Lock()
	for labelIdx := range dn.somaWeights {
		for i := range dn.somaWeights[labelIdx] {
			dn.somaWeights[labelIdx][i] += lr * somaWeightGrads[labelIdx][i]
		}
		dn.somaBias[labelIdx] += lr * somaBiasGrads[labelIdx]
	}
	for i := range dn.compartments {
		for j := range dn.compartments[i].weights {
			dn.compartments[i].weights[j] += lr * compWeightGrads[i][j]
		}
		dn.compartments[i].bias += lr * compBiasGrads[i]
	}
	dn.mu.Unlock()

	lossMu.Lock()
	*loss += localLoss
	lossMu.Unlock()
}

// --- Main ---
func main() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())

	fmt.Println("Generating training data for gates and adders...")
	allData := makeMultiLabelData()

	neuron := NewDendriticNeuron(3, 64, len(labels))

	fmt.Println("Training universal neuron on all logic gates and blocks...")
	epochs := 40000
	lr := 0.05

	for e := 0; e < epochs; e++ {
		var wg sync.WaitGroup
		totalLoss := 0.0
		var lossMu sync.Mutex

		chunkSize := len(allData) / runtime.NumCPU()
		for i := 0; i < runtime.NumCPU(); i++ {
			start := i * chunkSize
			end := start + chunkSize
			if i == runtime.NumCPU()-1 {
				end = len(allData)
			}
			wg.Add(1)
			go trainWorker(neuron, allData[start:end], lr, &wg, &totalLoss, &lossMu)
		}
		wg.Wait()

		avgLoss := totalLoss / float64(len(allData)*len(labels))
		if e%1000 == 0 {
			fmt.Printf("Epoch %d, Error: %.6f, LR: %.5f\n", e, avgLoss, lr)
		}

		// LR decay every 2000 epochs
		if e > 0 && e%2000 == 0 {
			lr *= 0.9
		}
	}

	fmt.Println("\nTesting trained neuron...\n")
	total, correct := 0, 0
	for _, d := range allData {
		preds, _ := neuron.Forward(d.inputs)
		for i, label := range labels {
			predBinary := math.Round(preds[i])
			ok := predBinary == d.expected[i]
			if ok {
				correct++
			}
			total++
			fmt.Printf("Label: %-18s Inputs: %v, Exp: %.0f, Pred: %.4f, R: %.0f, Correct: %v\n",
				label, d.inputs, d.expected[i], preds[i], predBinary, ok)
		}
	}
	acc := float64(correct) / float64(total) * 100
	fmt.Printf("\nOverall Accuracy: %.2f%%\n", acc)
}
