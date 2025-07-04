

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

// --- 1. Dendritic Compartment (Modified for Sparse Connections) ---
type DendriticCompartment struct {
	weights      []float64
	bias         float64
	inputIndices []int // NEW: To store which inputs this compartment connects to
}

// numTotalInputs: size of the previous layer
// numConnections: how many random inputs to connect to
func NewDendriticCompartment(numTotalInputs, numConnections int) *DendriticCompartment {
	if numConnections > numTotalInputs {
		numConnections = numTotalInputs
	}

	// Randomly select input indices without replacement
	p := rand.Perm(numTotalInputs)
	inputIndices := p[:numConnections]

	weights := make([]float64, numConnections)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1
	}

	return &DendriticCompartment{
		weights:      weights,
		bias:         rand.Float64()*2 - 1,
		inputIndices: inputIndices,
	}
}

func (dc *DendriticCompartment) Process(allInputs []float64) float64 {
	sum := dc.bias
	// Sum over the sparse connections only
	for i, weight := range dc.weights {
		inputIndex := dc.inputIndices[i]
		sum += allInputs[inputIndex] * weight
	}
	return math.Tanh(sum)
}

// --- 2. The Dendritic Neuron (Modified for Sparse Backprop) ---
type DendriticNeuron struct {
	compartments         []*DendriticCompartment
	somaWeights          []float64
	somaBias             float64
	lastBasalInputs      []float64
	lastCompartmentOutputs []float64
	lastSomaOutput       float64
}

// numInputsForThisLayer: size of the previous layer
// numCompartments: number of dendritic compartments in this neuron
// synapsesPerCompartment: number of sparse connections for each compartment
func NewDendriticNeuron(numInputsForThisLayer, numCompartments, synapsesPerCompartment int) *DendriticNeuron {
	compartments := make([]*DendriticCompartment, numCompartments)
	for i := 0; i < numCompartments; i++ {
		compartments[i] = NewDendriticCompartment(numInputsForThisLayer, synapsesPerCompartment)
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

// Train method is now simpler, as the complex backprop logic is in the dANN struct
func (dn *DendriticNeuron) Train(dLoss_dRawOutput float64, learningRate float64) {
	somaWeightGradients := make([]float64, len(dn.somaWeights))
	for i, compOutput := range dn.lastCompartmentOutputs {
		somaWeightGradients[i] = dLoss_dRawOutput * compOutput
	}
	somaBiasGradient := dLoss_dRawOutput

	for i, comp := range dn.compartments {
		errorCompartmentOutput := dLoss_dRawOutput * dn.somaWeights[i]
		// Gradient of Tanh is (1 - tanh^2(x))
		deltaCompartmentSum := errorCompartmentOutput * (1 - math.Pow(dn.lastCompartmentOutputs[i], 2))
		for j, weightIndex := range comp.inputIndices {
			// Update weights for the sparse connections
			comp.weights[j] -= learningRate * deltaCompartmentSum * dn.lastBasalInputs[weightIndex]
		}
		comp.bias -= learningRate * deltaCompartmentSum
	}

	for i := range dn.somaWeights {
		dn.somaWeights[i] -= learningRate * somaWeightGradients[i]
	}
	dn.somaBias -= learningRate * somaBiasGradient
}


// --- 3. The dANN (Modified for Feedback Alignment) ---
type dANN struct {
	layers          [][]*DendriticNeuron
	feedbackWeights [][][]float64 // NEW: Fixed random weights for backprop
	numInputs       int
	numOutputs      int
}

func NewdANN(layerSizes []int, numCompartmentsPerNeuron, synapsesPerCompartment int) *dANN {
	ann := &dANN{
		numInputs:  layerSizes[0],
		numOutputs: layerSizes[len(layerSizes)-1],
	}

	// Initialize layers with forward weights
	for i := 1; i < len(layerSizes); i++ {
		numNeuronsInLayer := layerSizes[i]
		numInputsForThisLayer := layerSizes[i-1]
		layer := make([]*DendriticNeuron, numNeuronsInLayer)
		for j := 0; j < numNeuronsInLayer; j++ {
			layer[j] = NewDendriticNeuron(numInputsForThisLayer, numCompartmentsPerNeuron, synapsesPerCompartment)
		}
		ann.layers = append(ann.layers, layer)
	}

	// Initialize fixed random feedback weights
	ann.feedbackWeights = make([][][]float64, len(ann.layers))
	for i := 1; i < len(ann.layers); i++ { // From layer 1 onwards
		prevLayerSize := layerSizes[i-1]
		currentLayerSize := layerSizes[i]
		ann.feedbackWeights[i] = make([][]float64, currentLayerSize)
		for j := 0; j < currentLayerSize; j++ {
			ann.feedbackWeights[i][j] = make([]float64, prevLayerSize)
			for k := 0; k < prevLayerSize; k++ {
				ann.feedbackWeights[i][j][k] = rand.Float64()*2 - 1 // Random weights [-1, 1]
			}
		}
	}

	return ann
}

// Forward pass is unchanged in logic, but uses the modified sparse neurons
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

// THE CORE CHANGE: Backpropagation with Feedback Alignment
func (ann *dANN) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		for _, d := range data {
			_, activatedOutputs := ann.Forward(d.inputs)
			finalNetworkOutput := activatedOutputs[len(ann.layers)-1]

			// Cross-entropy error calculation
			for i := range finalNetworkOutput {
				totalError -= d.expected[i] * math.Log(finalNetworkOutput[i]+1e-9)
			}

			// --- Biologically Plausible Backpropagation ---
			dError_dActivation := make([][]float64, len(ann.layers))
			outputLayerIdx := len(ann.layers) - 1

			// 1. Error at the output layer (this is directly known)
			dError_dActivation[outputLayerIdx] = make([]float64, len(finalNetworkOutput))
			for i := range finalNetworkOutput {
				// Derivative of CrossEntropy+Softmax is simple
				dError_dActivation[outputLayerIdx][i] = finalNetworkOutput[i] - d.expected[i]
			}

			// 2. Propagate error to hidden layers using FIXED RANDOM FEEDBACK WEIGHTS
			for l := len(ann.layers) - 2; l >= 0; l-- {
				nextLayerError := dError_dActivation[l+1]
				currentLayer := ann.layers[l]
				dError_dActivation[l] = make([]float64, len(currentLayer))

				for i := range currentLayer { // For each neuron in the current hidden layer
					errorSum := 0.0
					// Sum the errors from the next layer, weighted by the RANDOM feedback weights
					for j := range ann.layers[l+1] {
						errorSum += nextLayerError[j] * ann.feedbackWeights[l+1][j][i]
					}
					
					// Multiply by the derivative of the activation function (sigmoid)
					activatedOutput := activatedOutputs[l][i]
					derivativeOfSigmoid := activatedOutput * (1 - activatedOutput)
					dError_dActivation[l][i] = errorSum * derivativeOfSigmoid
				}
			}

			// 3. Update weights using the locally available (approximated) error signal
			for l := 0; l < len(ann.layers); l++ {
				for i, neuron := range ann.layers[l] {
					neuron.Train(dError_dActivation[l][i], learningRate)
				}
			}
		}
		if epoch > 0 && (epoch%100 == 0 || epoch == epochs-1) {
			fmt.Printf("Epoch %d, Avg Error: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- Training Data (Unchanged) ---
type TrainingData struct {
	inputs   []float64
	expected []float64 // One-hot encoded
}

// --- Data Loading (Unchanged, assuming tumor.csv is in the right path) ---
func loadTumorData(filePath string) ([]TrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 
	_, err = reader.Read()
	if err != nil {
		return nil, err
	}

	var rawData [][]float64
	var diagnoses []string

	diagnosisMap := map[string]int{"B": 0, "M": 1}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) != 32 {
			continue
		}

		var inputs []float64
		for i := 2; i < len(record); i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, val)
		}
		rawData = append(rawData, inputs)
		diagnoses = append(diagnoses, record[1])
	}

	numFeatures := len(rawData[0])
	minVals := make([]float64, numFeatures)
	maxVals := make([]float64, numFeatures)

	for i := 0; i < numFeatures; i++ {
		minVals[i] = rawData[0][i]
		maxVals[i] = rawData[0][i]
	}

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
				normalizedInputs[i] = 0.0
			} else {
				normalizedInputs[i] = (val - minVals[i]) / denominator
			}
		}

		diagnosis := diagnoses[idx]
		diagnosisIndex, ok := diagnosisMap[diagnosis]
		if !ok {
			continue
		}

		expected := make([]float64, 2)
		expected[diagnosisIndex] = 1.0

		normalizedData = append(normalizedData, TrainingData{inputs: normalizedInputs, expected: expected})
	}
	return normalizedData, nil
}

// --- Main Execution ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Loading Tumor Classification Training Data ---")
	tumorData, err := loadTumorData("C:/run/TestGemCli/dANN/tumor.csv")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d data points for Tumor Classification problem.\n\n", len(tumorData))

	rand.Shuffle(len(tumorData), func(i, j int) { tumorData[i], tumorData[j] = tumorData[j], tumorData[i] })
	splitIndex := int(float64(len(tumorData)) * 0.8)
	trainData := tumorData[:splitIndex]
	testData := tumorData[splitIndex:]

	// 30 inputs, 15 hidden neurons, 2 outputs
	// Each neuron has 8 compartments, each compartment connects to 10 random inputs
	ann := NewdANN([]int{30, 15, 2}, 8, 10) 
	fmt.Println("--- Training Bio-Plausible dANN on Tumor Classification problem ---")
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

		if predictedClass == trueClass {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testData)) * 100
	fmt.Printf("\nFinal Accuracy on Bio-Plausible Tumor Classification: %.2f%%\n", accuracy)
}
