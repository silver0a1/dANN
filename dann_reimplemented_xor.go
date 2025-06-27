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

// --- 2. The Dendritic Neuron (Corrected) ---
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

// THIS IS THE CRITICAL FIX
func (dn *DendriticNeuron) Train(dLoss_dRawOutput float64, learningRate float64) {
    // Create a temporary slice to hold the gradients for the soma weights
    somaWeightGradients := make([]float64, len(dn.somaWeights))
    for i, compOutput := range dn.lastCompartmentOutputs {
        somaWeightGradients[i] = dLoss_dRawOutput * compOutput
    }
    somaBiasGradient := dLoss_dRawOutput

    // Update Compartment weights and bias USING THE OLD SOMA WEIGHTS
    for i, comp := range dn.compartments {
        // Propagate error using the soma weight from *before* the update
        errorCompartmentOutput := dLoss_dRawOutput * dn.somaWeights[i]
        deltaCompartmentSum := errorCompartmentOutput * (1 - math.Pow(dn.lastCompartmentOutputs[i], 2))
        for j, input := range dn.lastBasalInputs {
            comp.weights[j] -= learningRate * deltaCompartmentSum * input
        }
        comp.bias -= learningRate * deltaCompartmentSum
    }

    // NOW, update the soma weights and bias using the stored gradients
    for i := range dn.somaWeights {
        dn.somaWeights[i] -= learningRate * somaWeightGradients[i]
    }
    dn.somaBias -= learningRate * somaBiasGradient
}


// calculateEffectiveWeight computes the effective weight of a single input connection
// on the neuron's raw soma output. This is crucial for backpropagation.
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

// --- 3. The dANN (Corrected) ---
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
	for i := 0; i < len(layerSizes); i++ {
		numNeuronsInLayer := layerSizes[i]
		var numInputsForThisLayer int
		if i == 0 {
			numInputsForThisLayer = ann.numInputs
		} else {
			numInputsForThisLayer = layerSizes[i-1]
		}
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
		for j, neuron := range layer {
			rawOutput := neuron.Forward(currentInputs)
			rawOutputs[i][j] = rawOutput
			if i < len(ann.layers)-1 {
				activatedOutputs[i][j] = 1.0 / (1.0 + math.Exp(-rawOutput))
			} else {
				activatedOutputs[i][j] = 1.0 / (1.0 + math.Exp(-rawOutput)) // Also activate output layer for prediction
			}
		}
		currentInputs = activatedOutputs[i]
	}
	return rawOutputs, activatedOutputs
}

func (ann *dANN) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for _, d := range data {
			_, activatedOutputs := ann.Forward(d.inputs)
			finalNetworkOutput := activatedOutputs[len(ann.layers)-1]
			for i := range finalNetworkOutput {
				err := d.expected - finalNetworkOutput[i]
				totalError += err * err
			}
			dLoss_dRawOutput := make([][]float64, len(ann.layers))
			outputLayerIdx := len(ann.layers) - 1
			outputLayer := ann.layers[outputLayerIdx]
			dLoss_dRawOutput[outputLayerIdx] = make([]float64, len(outputLayer))
			for i := range outputLayer {
				errorWithRespectToActivatedOutput := -(d.expected - finalNetworkOutput[i])
				derivativeOfSigmoid := finalNetworkOutput[i] * (1 - finalNetworkOutput[i])
				dLoss_dRawOutput[outputLayerIdx][i] = errorWithRespectToActivatedOutput * derivativeOfSigmoid
			}
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
			for l := 0; l < len(ann.layers); l++ {
				for i, neuron := range ann.layers[l] {
					neuron.Train(dLoss_dRawOutput[l][i], learningRate)
				}
			}
		}
		if epoch > 0 && (epoch%5000 == 0 || epoch == epochs-1) {
			fmt.Printf("Epoch %d, Avg Error: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- Training Data (Unchanged) ---
type TrainingData struct {
	inputs   []float64
	expected float64
}

func generateXORData(numPoints int) []TrainingData {
	data := make([]TrainingData, numPoints)
	for i := 0; i < numPoints; i++ {
		x1 := float64(rand.Intn(2))
		x2 := float64(rand.Intn(2))
		expected := 0.0
		if (x1 == 0 && x2 == 1) || (x1 == 1 && x2 == 0) {
			expected = 1.0
		}
		data[i] = TrainingData{inputs: []float64{x1, x2}, expected: expected}
	}
	return data
}

// --- Main Execution for dANN ---
func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("--- Generating XOR Training Data for dANN ---")
	xorData := generateXORData(200) // Using more data points
	fmt.Printf("Generated %d data points for XOR problem.\n\n", len(xorData))
	ann := NewdANN([]int{2, 4, 1}, 8)
	fmt.Println("--- Training dANN on XOR problem ---")
	ann.Train(xorData, 30000, 0.05) // Fewer epochs, higher learning rate now that it's stable
	fmt.Println("\n--- Training Complete ---")
	fmt.Println("\n--- Testing Trained dANN ---")
	correct := 0
    testData := generateXORData(100) // Test on a separate set
	for _, d := range testData {
		_, activatedOutputs := ann.Forward(d.inputs)
		prediction := activatedOutputs[len(activatedOutputs)-1][0]
		roundedPrediction := math.Round(prediction)
		isCorrect := roundedPrediction == d.expected
		if isCorrect {
			correct++
		}
		fmt.Printf("Input: [%.0f, %.0f], Expected: %.0f, Prediction: %.4f, Rounded: %.0f, Correct: %t\n",
			d.inputs[0], d.inputs[1], d.expected, prediction, roundedPrediction, isCorrect)
	}
	accuracy := float64(correct) / float64(len(testData)) * 100
	fmt.Printf("\nFinal Accuracy on XOR: %.2f%%\n", accuracy)
}