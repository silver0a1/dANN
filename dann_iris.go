package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
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
		weights[i] = rand.Float64()*2 - 1 // Random weights between -1 and 1
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
	return math.Tanh(sum) // Tanh activation for compartments
}

// --- 2. The Dendritic Neuron (Unchanged) ---
type DendriticNeuron struct {
	compartments         []*DendriticCompartment
	somaWeights          []float64
	somaBias             float64
	lastBasalInputs      []float64
	lastCompartmentOutputs []float64
	lastSomaOutput       float64
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

func (dn *DendriticNeuron) Forward(inputs []float64) float64 {
	dn.lastBasalInputs = inputs
	dn.lastCompartmentOutputs = make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		dn.lastCompartmentOutputs[i] = compartment.Process(inputs)
	}
	finalSum := dn.somaBias
	for i, output := range dn.lastCompartmentOutputs {
		finalSum += output * dn.somaWeights[i]
	}
	dn.lastSomaOutput = finalSum
	return finalSum
}

func (dn *DendriticNeuron) Train(dLoss_dRawOutput float64, learningRate float64) {
    somaWeightGradients := make([]float64, len(dn.somaWeights))
    for i, compOutput := range dn.lastCompartmentOutputs {
        somaWeightGradients[i] = dLoss_dRawOutput * compOutput
    }
    somaBiasGradient := dLoss_dRawOutput

    for i, comp := range dn.compartments {
        errorCompartmentOutput := dLoss_dRawOutput * dn.somaWeights[i]
        deltaCompartmentSum := errorCompartmentOutput * (1 - math.Pow(dn.lastCompartmentOutputs[i], 2))
        for j, input := range dn.lastBasalInputs {
            comp.weights[j] -= learningRate * deltaCompartmentSum * input
        }
        comp.bias -= learningRate * deltaCompartmentSum
    }

    for i := range dn.somaWeights {
        dn.somaWeights[i] -= learningRate * somaWeightGradients[i]
    }
    dn.somaBias -= learningRate * somaBiasGradient
}

func (dn *DendriticNeuron) calculateEffectiveWeight(inputIndex int) float64 {
	effectiveWeight := 0.0
	for i, compartment := range dn.compartments {
		dCompOutput_dCompSum := 1 - math.Pow(dn.lastCompartmentOutputs[i], 2)
		compWeightForInput := compartment.weights[inputIndex]
		somaWeightForComp := dn.somaWeights[i]
		effectiveWeight += somaWeightForComp * dCompOutput_dCompSum * compWeightForInput
	}
	return effectiveWeight
}

// --- 3. The dANN (Modified for Multi-Class) ---
type dANN struct {
	layers     [][]*DendriticNeuron
	numInputs  int
	numOutputs int
}

func NewdANN(layerSizes []int, numCompartmentsPerNeuron int) *dANN {
	ann := &dANN{
		numInputs:  layerSizes[0],
		numOutputs: layerSizes[len(layerSizes)-1],
	}
	for i := 1; i < len(layerSizes); i++ {
		numNeuronsInLayer := layerSizes[i]
		numInputsForThisLayer := layerSizes[i-1]
		layer := make([]*DendriticNeuron, numNeuronsInLayer)
		for j := 0; j < numNeuronsInLayer; j++ {
			layer[j] = NewDendriticNeuron(numInputsForThisLayer, numCompartmentsPerNeuron)
		}
		ann.layers = append(ann.layers, layer)
	}
	return ann
}

func (ann *dANN) Forward(inputs []float64) ([][]float64, [][]float64) {
	rawOutputs := make([][]float64, len(ann.layers))
	activatedOutputs := make([][]float64, len(ann.layers))
	currentInputs := inputs

	for i, layer := range ann.layers {
		rawOutputs[i] = make([]float64, len(layer))
		activatedOutputs[i] = make([]float64, len(layer))
		layerInputs := currentInputs
		for j, neuron := range layer {
			rawOutput := neuron.Forward(layerInputs)
			rawOutputs[i][j] = rawOutput
			if i < len(ann.layers)-1 {
				activatedOutputs[i][j] = 1.0 / (1.0 + math.Exp(-rawOutput)) // Sigmoid for hidden layers
			}
		}
		if i == len(ann.layers)-1 { // Softmax for output layer
			activatedOutputs[i] = softmax(rawOutputs[i])
		}
		currentInputs = activatedOutputs[i]
	}
	return rawOutputs, activatedOutputs
}

func softmax(raw []float64) []float64 {
	max := raw[0]
	for _, v := range raw {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	out := make([]float64, len(raw))
	for i, v := range raw {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func (ann *dANN) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		for _, d := range data {
			_, activatedOutputs := ann.Forward(d.inputs)
			finalNetworkOutput := activatedOutputs[len(ann.layers)-1]

			// Cross-entropy error calculation
			for i := range finalNetworkOutput {
				totalError -= d.expected[i] * math.Log(finalNetworkOutput[i]+1e-9) // Add small epsilon to avoid log(0)
			}

			// Backpropagation for Softmax and Cross-Entropy
			dLoss_dRawOutput := make([][]float64, len(ann.layers))
			outputLayerIdx := len(ann.layers) - 1
			outputLayer := ann.layers[outputLayerIdx]
			dLoss_dRawOutput[outputLayerIdx] = make([]float64, len(outputLayer))

			for i := range outputLayer {
				dLoss_dRawOutput[outputLayerIdx][i] = finalNetworkOutput[i] - d.expected[i]
			}

			// Backpropagate through hidden layers
			for l := len(ann.layers) - 2; l >= 0; l-- {
				currentLayer := ann.layers[l]
				nextLayer := ann.layers[l+1]
				dLoss_dRawOutput[l] = make([]float64, len(currentLayer))
				for i := range currentLayer {
					errorSum := 0.0
					for j, nextNeuron := range nextLayer {
						effectiveWeight := nextNeuron.calculateEffectiveWeight(i)
						errorSum += dLoss_dRawOutput[l+1][j] * effectiveWeight
					}
					activatedOutput := activatedOutputs[l][i]
					derivativeOfSigmoid := activatedOutput * (1 - activatedOutput)
					dLoss_dRawOutput[l][i] = errorSum * derivativeOfSigmoid
				}
			}

			// Update weights
			for l := 0; l < len(ann.layers); l++ {
				for i, neuron := range ann.layers[l] {
					neuron.Train(dLoss_dRawOutput[l][i], learningRate)
				}
			}
		}
		if epoch > 0 && (epoch%100 == 0 || epoch == epochs-1) {
			fmt.Printf("Epoch %d, Avg Error: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- Training Data (Modified for Iris) ---
type TrainingData struct {
	inputs   []float64
	expected []float64 // One-hot encoded
}

func loadIrisData(filePath string) ([]TrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []TrainingData

	speciesMap := map[string]int{
		"Iris-setosa":     0,
		"Iris-versicolor": 1,
		"Iris-virginica":  2,
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) != 5 {
			continue // Skip malformed lines
		}

		var inputs []float64
		for i := 0; i < 4; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, val)
		}

		species := record[4]
		speciesIndex, ok := speciesMap[species]
		if !ok {
			continue // Skip unknown species
		}

		expected := make([]float64, 3)
		expected[speciesIndex] = 1.0

		data = append(data, TrainingData{inputs: inputs, expected: expected})
	}
	return data, nil
}

// --- Main Execution for dANN on Iris ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Loading Iris Training Data ---")
	irisData, err := loadIrisData("C:/run/TestGemCli/dANN/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d data points for Iris problem.\n\n", len(irisData))

	// Split data into training and testing sets (80/20 split)
	rand.Shuffle(len(irisData), func(i, j int) { irisData[i], irisData[j] = irisData[j], irisData[i] })
	splitIndex := int(float64(len(irisData)) * 0.8)
	trainData := irisData[:splitIndex]
	testData := irisData[splitIndex:]

	ann := NewdANN([]int{4, 8, 3}, 8) // 4 inputs, 8 hidden neurons, 3 outputs
	fmt.Println("--- Training dANN on Iris problem ---")
	ann.Train(trainData, 10000, 0.01)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained dANN on Iris ---")
	correct := 0
	for _, d := range testData {
		_, activatedOutputs := ann.Forward(d.inputs)
		prediction := activatedOutputs[len(activatedOutputs)-1]

		predictedClass := 0
		maxProb := prediction[0]
		for i, p := range prediction {
			if p > maxProb {
				maxProb = p
				predictedClass = i
			}
		}

		trueClass := 0
		for i, v := range d.expected {
			if v == 1.0 {
				trueClass = i
				break
			}
		}

		if predictedClass == trueClass {
			correct++
		}
		fmt.Printf("Input: %v, Expected: %d, Prediction: %d (Probs: %v)\n", d.inputs, trueClass, predictedClass, prediction)
	}
	accuracy := float64(correct) / float64(len(testData)) * 100
	fmt.Printf("\nFinal Accuracy on Iris: %.2f%%\n", accuracy)
}