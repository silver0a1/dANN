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

// --- Training Data ---
type TrainingData struct {
	inputs   []float64
	expected []float64 // One-hot encoded
}

func loadTumorData(filePath string) ([]TrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // Allow variable number of fields per record
	// Skip header row
	_, err = reader.Read()
	if err != nil {
		return nil, err
	}

	var rawData [][]float64
	var diagnoses []string

	diagnosisMap := map[string]int{
		"B": 0, // Benign
		"M": 1, // Malignant
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		// tumor.csv has 32 columns: id, diagnosis, and 30 features
		if len(record) != 32 {
			continue // Skip malformed lines
		}

		var inputs []float64
		// Start from index 2 to skip 'id' and 'diagnosis'
		for i := 2; i < len(record); i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, val)
		}
		rawData = append(rawData, inputs)
		diagnoses = append(diagnoses, record[1]) // 'diagnosis' is at index 1
	}

	// Normalize data (Min-Max Scaling)
	numFeatures := len(rawData[0])
	minVals := make([]float64, numFeatures)
	maxVals := make([]float64, numFeatures)

	// Initialize min/max with first data point
	for i := 0; i < numFeatures; i++ {
		minVals[i] = rawData[0][i]
		maxVals[i] = rawData[0][i]
	}

	// Find min and max for each feature
	for _, row := range rawData {
		for i, val := range row {
			if val < minVals[i] {
				minVals[i] = val
			}
			if val > maxVals[i] {
				maxVals[i] = val
			}
		}
	}

	var normalizedData []TrainingData
	for idx, inputs := range rawData {
		normalizedInputs := make([]float64, numFeatures)
		for i, val := range inputs {
			denominator := maxVals[i] - minVals[i]
			if denominator == 0 {
				normalizedInputs[i] = 0.0 // Avoid division by zero if all values are the same
			} else {
				normalizedInputs[i] = (val - minVals[i]) / denominator
			}
		}

		diagnosis := diagnoses[idx]
		diagnosisIndex, ok := diagnosisMap[diagnosis]
		if !ok {
			continue // Skip unknown diagnosis
		}

		expected := make([]float64, 2) // 2 output classes: Benign, Malignant
		expected[diagnosisIndex] = 1.0

		normalizedData = append(normalizedData, TrainingData{inputs: normalizedInputs, expected: expected})
	}
	return normalizedData, nil
}

// --- Main Execution for dANN on Tumor Classification ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Loading Tumor Classification Training Data ---")
	tumorData, err := loadTumorData("C:/run/TestGemCli/dANN/tumor.csv")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d data points for Tumor Classification problem.\n\n", len(tumorData))

	// Split data into training and testing sets (80/20 split)
	rand.Shuffle(len(tumorData), func(i, j int) { tumorData[i], tumorData[j] = tumorData[j], tumorData[i] })
	splitIndex := int(float64(len(tumorData)) * 0.8)
	trainData := tumorData[:splitIndex]
	testData := tumorData[splitIndex:]

	// 30 inputs (features), 15 hidden neurons (example), 2 outputs (Benign/Malignant)
	ann := NewdANN([]int{30, 15, 2}, 8) 
	fmt.Println("--- Training dANN on Tumor Classification problem ---")
	ann.Train(trainData, 1000, 0.01)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained dANN on Tumor Classification ---")
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

		// Map back to 'B' or 'M' for display
		predictedLabel := "B"
		if predictedClass == 1 {
			predictedLabel = "M"
		}
		trueLabel := "B"
		if trueClass == 1 {
			trueLabel = "M"
		}

		if predictedClass == trueClass {
			correct++
		}
		fmt.Printf("Input (first 5 features): %v, Expected: %s, Prediction: %s (Probs: %v)\n", d.inputs[:5], trueLabel, predictedLabel, prediction)
	}
	accuracy := float64(correct) / float64(len(testData)) * 100
	fmt.Printf("\nFinal Accuracy on Tumor Classification: %.2f%%\n", accuracy)
}
