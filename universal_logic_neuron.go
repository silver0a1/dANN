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
		w[i] = rand.Float64()*0.1 - 0.05 // smaller init weights for stability
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

// --- Dendritic Neuron ---
type DendriticNeuron struct {
	compartments []*DendriticCompartment
	somaWeights  []float64
	somaBias     float64
	labelWeights map[string]float64
	mu           sync.RWMutex
}

func NewDendriticNeuron(numInputs, numCompartments int) *DendriticNeuron {
	comps := make([]*DendriticCompartment, numCompartments)
	for i := range comps {
		comps[i] = NewDendriticCompartment(numInputs)
	}
	somaW := make([]float64, numCompartments)
	for i := range somaW {
		somaW[i] = rand.Float64()*0.1 - 0.05
	}
	return &DendriticNeuron{
		compartments: comps,
		somaWeights:  somaW,
		somaBias:     rand.Float64()*0.1 - 0.05,
		labelWeights: make(map[string]float64),
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (dn *DendriticNeuron) Forward(inputs []float64, label string) (float64, []float64) {
	compOuts := make([]float64, len(dn.compartments))
	for i := range dn.compartments {
		compOuts[i] = dn.compartments[i].Process(inputs)
	}
	sum := dn.somaBias
	for i, o := range compOuts {
		sum += o * dn.somaWeights[i]
	}

	dn.mu.RLock()
	labelWeight := dn.labelWeights[label]
	dn.mu.RUnlock()

	sum += labelWeight
	return sigmoid(sum), compOuts
}

type TrainingData struct {
	inputs   []float64
	expected float64
	label    string
}

// --- Input normalization ---
// Identity: keep inputs 0 or 1 as is
func normalizeInput(x float64) float64 {
	return x
}

// --- Training with batch gradient accumulation and concurrency ---
func trainWorker(dn *DendriticNeuron, samples []TrainingData, lr float64, wg *sync.WaitGroup, loss *float64, lossMu *sync.Mutex) {
	defer wg.Done()

	localLoss := 0.0

	somaWeightGrad := make([]float64, len(dn.somaWeights))
	var somaBiasGrad float64
	labelWeightGrad := make(map[string]float64)
	compWeightGrad := make([][]float64, len(dn.compartments))
	compBiasGrad := make([]float64, len(dn.compartments))
	for i := range dn.compartments {
		compWeightGrad[i] = make([]float64, len(dn.compartments[i].weights))
	}

	for _, d := range samples {
		pred, outs := dn.Forward(d.inputs, d.label)
		err := d.expected - pred
		localLoss += err * err
		delta := err * pred * (1 - pred)

		for i, o := range outs {
			somaWeightGrad[i] += delta * o
		}
		somaBiasGrad += delta
		labelWeightGrad[d.label] += delta

		for i := range dn.compartments {
			errC := delta * dn.somaWeights[i]
			deltaC := errC * (1 - outs[i]*outs[i])
			for j, x := range d.inputs {
				compWeightGrad[i][j] += deltaC * x
			}
			compBiasGrad[i] += deltaC
		}
	}

	dn.mu.Lock()
	for i := range dn.somaWeights {
		dn.somaWeights[i] += lr * somaWeightGrad[i]
	}
	dn.somaBias += lr * somaBiasGrad
	for label, grad := range labelWeightGrad {
		dn.labelWeights[label] += lr * grad
	}
	for i := range dn.compartments {
		for j := range dn.compartments[i].weights {
			dn.compartments[i].weights[j] += lr * compWeightGrad[i][j]
		}
		dn.compartments[i].bias += lr * compBiasGrad[i]
	}
	dn.mu.Unlock()

	lossMu.Lock()
	*loss += localLoss
	lossMu.Unlock()
}

// --- Generate training data for logic gates and blocks ---
func makeLogicGateData() []TrainingData {
	data := []TrainingData{}
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for _, in := range inputs {
		a, b := normalizeInput(in[0]), normalizeInput(in[1])
		data = append(data, TrainingData{[]float64{a, b}, a * b, "AND"})
		data = append(data, TrainingData{[]float64{a, b}, math.Max(a, b), "OR"})
		data = append(data, TrainingData{[]float64{a, b}, 1 - a, "NOT_A"})
		data = append(data, TrainingData{[]float64{a, b}, 1 - b, "NOT_B"})
		data = append(data, TrainingData{[]float64{a, b}, 1 - a*b, "NAND"})
		data = append(data, TrainingData{[]float64{a, b}, 1 - math.Max(a, b), "NOR"})
		data = append(data, TrainingData{[]float64{a, b}, float64(int(a) ^ int(b)), "XOR"})
		data = append(data, TrainingData{[]float64{a, b}, 1 - float64(int(a)^int(b)), "XNOR"})
		data = append(data, TrainingData{[]float64{a, b}, a, "BUFFER_A"})
		data = append(data, TrainingData{[]float64{a, b}, b, "BUFFER_B"})
		data = append(data, TrainingData{[]float64{a, b}, float64(int(a) ^ int(b)), "HALF_ADDER_SUM"})
		data = append(data, TrainingData{[]float64{a, b}, a * b, "HALF_ADDER_CARRY"})
	}
	return data
}

func makeFullAdderData() []TrainingData {
	data := []TrainingData{}
	for _, in := range [][]float64{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
		{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
	} {
		a, b, cin := normalizeInput(in[0]), normalizeInput(in[1]), normalizeInput(in[2])
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

	neuron := NewDendriticNeuron(3, 64) // 3 inputs max, 64 compartments

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

		avgLoss := totalLoss / float64(len(allData))
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
