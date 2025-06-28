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

// --- 2. The Dendritic Neuron ---
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

// --- 3. The dANN (Modified for Regression) ---
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
			if i < len(ann.layers)-1 { // Sigmoid for hidden layers
				activatedOutputs[i][j] = 1.0 / (1.0 + math.Exp(-rawOutput))
			} else { // Linear activation for output layer (regression)
				activatedOutputs[i][j] = rawOutput
			}
		}
		currentInputs = activatedOutputs[i]
	}
	return rawOutputs, activatedOutputs
}

func (ann *dANN) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		for _, d := range data {
			_, activatedOutputs := ann.Forward(d.inputs)
			finalNetworkOutput := activatedOutputs[len(ann.layers)-1]

			// Mean Squared Error calculation
			for i := range finalNetworkOutput {
				totalError += math.Pow(d.expected[i]-finalNetworkOutput[i], 2)
			}

			// Backpropagation for MSE
			dLoss_dRawOutput := make([][]float64, len(ann.layers))
			outputLayerIdx := len(ann.layers)-1
			outputLayer := ann.layers[outputLayerIdx]
			dLoss_dRawOutput[outputLayerIdx] = make([]float64, len(outputLayer))

			for i := range outputLayer {
				dLoss_dRawOutput[outputLayerIdx][i] = 2 * (finalNetworkOutput[i] - d.expected[i]) // Derivative of MSE
			}

			// Backpropagate through hidden layers
			for l := len(ann.layers)-2; l >= 0; l-- {
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
			fmt.Printf("Epoch %d, Avg MSE: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- Training Data ---
type TrainingData struct {
	inputs   []float64
	expected []float64 // For regression, this will be a single value
}

// MinMaxScaler stores min and max values for scaling
type MinMaxScaler struct {
	min []float64
	max []float64
}

// NewMinMaxScaler initializes the scaler with min/max values from data
func NewMinMaxScaler(data [][]float64) *MinMaxScaler {
	numFeatures := len(data[0])
	minVals := make([]float64, numFeatures)
	maxVals := make([]float64, numFeatures)

	for i := 0; i < numFeatures; i++ {
		minVals[i] = data[0][i]
		maxVals[i] = data[0][i]
	}

	for _, row := range data {
		for i, val := range row {
			if val < minVals[i] {
				minVals[i] = val
			}
			if val > maxVals[i] {
				maxVals[i] = val
			}
		}
	}
	return &MinMaxScaler{min: minVals, max: maxVals}
}

// Transform scales the input values
func (s *MinMaxScaler) Transform(inputs []float64) []float64 {
	scaledInputs := make([]float64, len(inputs))
	for i, val := range inputs {
		denominator := s.max[i] - s.min[i]
		if denominator == 0 {
			scaledInputs[i] = 0.0
		} else {
			scaledInputs[i] = (val - s.min[i]) / denominator
		}
	}
	return scaledInputs
}

// InverseTransform scales the output value back to original range
func (s *MinMaxScaler) InverseTransform(scaledValue float64, featureIndex int) float64 {
	return scaledValue*(s.max[featureIndex]-s.min[featureIndex]) + s.min[featureIndex]
}

func loadSineData(numPoints int, xMin, xMax, yMin, yMax float64) ([]TrainingData, *MinMaxScaler, *MinMaxScaler) {
	var rawInputs [][]float64
	var rawTargets [][]float64

	for i := 0; i < numPoints; i++ {
		x := xMin + rand.Float64()*(xMax-xMin)
		y := yMin + rand.Float64()*(yMax-yMin)
		// The target function: sin(x * y)
		target := math.Sin(x * y)

		rawInputs = append(rawInputs, []float64{x, y})
		rawTargets = append(rawTargets, []float64{target})
	}

	inputScaler := NewMinMaxScaler(rawInputs)
	targetScaler := NewMinMaxScaler(rawTargets)

	var data []TrainingData
	for i := range rawInputs {
		scaledInputs := inputScaler.Transform(rawInputs[i])
		scaledTarget := targetScaler.Transform(rawTargets[i])
		data = append(data, TrainingData{inputs: scaledInputs, expected: scaledTarget})
	}
	return data, inputScaler, targetScaler
}

// --- Main Execution for dANN on Sine Approximation ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Generating Sine Approximation Data ---")
	numDataPoints := 2000 // Number of data points for training and testing
	xMin, xMax := -5.0, 5.0
	yMin, yMax := -5.0, 5.0
	sineData, _, targetScaler := loadSineData(numDataPoints, xMin, xMax, yMin, yMax)
	fmt.Printf("Generated %d data points for Sine Approximation problem.\n\n", len(sineData))

	// Split data into training and testing sets (80/20 split)
	rand.Shuffle(len(sineData), func(i, j int) { sineData[i], sineData[j] = sineData[j], sineData[i] })
	splitIndex := int(float64(len(sineData)) * 0.8)
	trainData := sineData[:splitIndex]
	testData := sineData[splitIndex:]

	// 2 inputs (x, y), 16 hidden neurons (example), 1 output (sin(x*y))
	ann := NewdANN([]int{2, 32, 1}, 16) 
	fmt.Println("--- Training dANN on Sine Approximation problem ---")
	ann.Train(trainData, 20000, 0.005) // Increased epochs, neurons, compartments, and adjusted learning rate for complex function
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained dANN on Sine Approximation ---")
	totalSquaredError := 0.0
	for _, d := range testData {
		_, activatedOutputs := ann.Forward(d.inputs)
		predictionScaled := activatedOutputs[len(activatedOutputs)-1][0]
		expectedScaled := d.expected[0]

		// Inverse transform to get original scale values for display and error calculation
		predictionOriginal := targetScaler.InverseTransform(predictionScaled, 0)
		expectedOriginal := targetScaler.InverseTransform(expectedScaled, 0)

		totalSquaredError += math.Pow(expectedOriginal-predictionOriginal, 2)

		// Print a sample of predictions
		// fmt.Printf("Input: [%.2f, %.2f], Expected: %.4f, Predicted: %.4f (Scaled Expected: %.4f, Scaled Predicted: %.4f)\n", 
		// 	inputScaler.InverseTransform(d.inputs[0], 0), 
		// 	inputScaler.InverseTransform(d.inputs[1], 1), 
		// 	expectedOriginal, predictionOriginal, expectedScaled, predictionScaled)
	}
	rmse := math.Sqrt(totalSquaredError / float64(len(testData)))
	fmt.Printf("\nFinal RMSE on Sine Approximation: %.4f\n", rmse)

	// Calculate R-squared
	var sumOfSquaresTotal float64
	var meanExpected float64
	for _, d := range testData {
		meanExpected += targetScaler.InverseTransform(d.expected[0], 0)
	}
	meanExpected /= float64(len(testData))

	for _, d := range testData {
		sumOfSquaresTotal += math.Pow(targetScaler.InverseTransform(d.expected[0], 0) - meanExpected, 2)
	}

	R2 := 1 - (totalSquaredError / sumOfSquaresTotal)
	fmt.Printf("Final R-squared on Sine Approximation: %.4f\n", R2)
}