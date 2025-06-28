package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- 1. Neuron (Standard ANN) ---
type Neuron struct {
	weights []float64
	bias    float64
	output  float64 // Output after activation
	delta   float64 // Error signal
}

func NewNeuron(numInputs int) *Neuron {
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = rand.NormFloat64() * math.Sqrt(2.0/float64(numInputs)) // He initialization
	}
	return &Neuron{
		weights: weights,
		bias:    rand.Float64()*2 - 1,
	}
}

// --- 2. Layer (Standard ANN) ---
type Layer struct {
	neurons []*Neuron
}

func NewLayer(numNeurons, numInputsPerNeuron int) *Layer {
	neurons := make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		neurons[i] = NewNeuron(numInputsPerNeuron)
	}
	return &Layer{neurons: neurons}
}

func (l *Layer) forward(inputs []float64, activationFunction func(float64) float64) []float64 {
	outputs := make([]float64, len(l.neurons))
	for i, neuron := range l.neurons {
		sum := neuron.bias
		for j, input := range inputs {
			sum += input * neuron.weights[j]
		}
		neuron.output = activationFunction(sum)
		outputs[i] = neuron.output
	}
	return outputs
}

// --- 3. The ANN (Standard Feedforward) ---
type ANN struct {
	layers []*Layer
}

func NewANN(layerSizes []int) *ANN {
	ann := &ANN{}
	for i := 1; i < len(layerSizes); i++ {
		numNeuronsInLayer := layerSizes[i]
		numInputsForThisLayer := layerSizes[i-1]
		layer := NewLayer(numNeuronsInLayer, numInputsForThisLayer)
		ann.layers = append(ann.layers, layer)
	}
	return ann
}

func (ann *ANN) Forward(inputs []float64) []float64 {
	currentInputs := inputs
	for i, layer := range ann.layers {
		var activation func(float64) float64
		if i < len(ann.layers)-1 {
			activation = sigmoid // Sigmoid for hidden layers
		} else {
			activation = linear // Linear for output layer
		}
		currentInputs = layer.forward(currentInputs, activation)
	}
	return currentInputs
}

func (ann *ANN) Train(data []TrainingData, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

		for _, d := range data {
			// Forward pass
			finalOutputs := ann.Forward(d.inputs)

			// Calculate error
			for i := range finalOutputs {
				totalError += math.Pow(d.expected[i]-finalOutputs[i], 2)
			}

			// Backward pass (Backpropagation)
			outputLayer := ann.layers[len(ann.layers)-1]
			for i, neuron := range outputLayer.neurons {
				error := d.expected[i] - neuron.output
				neuron.delta = error * linearDerivative(neuron.output)
			}

			for l := len(ann.layers) - 2; l >= 0; l-- {
				currentLayer := ann.layers[l]
				nextLayer := ann.layers[l+1]
				for i, neuron := range currentLayer.neurons {
					var sum float64
					for _, nextNeuron := range nextLayer.neurons {
						sum += nextNeuron.weights[i] * nextNeuron.delta
					}
					neuron.delta = sum * sigmoidDerivative(neuron.output)
				}
			}

			// Update weights
			inputs := d.inputs
			for _, layer := range ann.layers {
				for _, neuron := range layer.neurons {
					for i, input := range inputs {
						neuron.weights[i] += learningRate * neuron.delta * input
					}
					neuron.bias += learningRate * neuron.delta
				}
				inputs = layer.forward(inputs, sigmoid) // Need outputs for next layer update
			}
		}

		if epoch > 0 && (epoch%100 == 0 || epoch == epochs-1) {
			fmt.Printf("Epoch %d, Avg MSE: %f\n", epoch, totalError/float64(len(data)))
		}
	}
}

// --- Activation Functions ---
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func linear(x float64) float64 {
	return x
}

func linearDerivative(x float64) float64 {
	return 1.0
}

// --- Training Data ---
type TrainingData struct {
	inputs   []float64
	expected []float64
}

// MinMaxScaler stores min and max values for scaling
type MinMaxScaler struct {
	min []float64
	max []float64
}

// NewMinMaxScaler initializes the scaler with min/max values from data
func NewMinMaxScaler(data [][]float64) *MinMaxScaler {
	if len(data) == 0 || len(data[0]) == 0 {
		return &MinMaxScaler{}
	}
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

// --- Main Execution for ANN on Sine Approximation ---
func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("--- Generating Sine Approximation Data ---")
	numDataPoints := 2000
	xMin, xMax := -5.0, 5.0
	yMin, yMax := -5.0, 5.0
	sineData, _, targetScaler := loadSineData(numDataPoints, xMin, xMax, yMin, yMax)
	fmt.Printf("Generated %d data points for Sine Approximation problem.\n\n", len(sineData))

	rand.Shuffle(len(sineData), func(i, j int) { sineData[i], sineData[j] = sineData[j], sineData[i] })
	splitIndex := int(float64(len(sineData)) * 0.8)
	trainData := sineData[:splitIndex]
	testData := sineData[splitIndex:]

	// 2 inputs (x, y), 32 hidden neurons, 1 output (sin(x*y))
	ann := NewANN([]int{2, 32, 1})
	fmt.Println("--- Training ANN on Sine Approximation problem ---")
	ann.Train(trainData, 20000, 0.005)
	fmt.Println("\n--- Training Complete ---")

	fmt.Println("\n--- Testing Trained ANN on Sine Approximation ---")
	totalSquaredError := 0.0
	for _, d := range testData {
		predictionScaled := ann.Forward(d.inputs)[0]
		expectedScaled := d.expected[0]

		predictionOriginal := targetScaler.InverseTransform(predictionScaled, 0)
		expectedOriginal := targetScaler.InverseTransform(expectedScaled, 0)

		totalSquaredError += math.Pow(expectedOriginal-predictionOriginal, 2)
	}
	rmse := math.Sqrt(totalSquaredError / float64(len(testData)))
	fmt.Printf("\nFinal RMSE on Sine Approximation: %.4f\n", rmse)

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
