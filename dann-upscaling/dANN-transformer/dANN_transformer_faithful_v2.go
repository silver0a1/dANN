
package main

import (
	"fmt"
	"math"
	"math/rand"
)

// --- Utility & Base Interface ---

func newMatrix(rows, cols int) []float64 {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(1.0/float64(cols))
	}
	return data
}

// Module defines the interface for a learnable component of the network.
// Note: The Forward/Backward signatures are simplified for this blueprint.
// A real implementation might need to handle different numbers of inputs/outputs.
type Module interface {
	// Forward pass through the module.
	Forward(input ...[][]float64) [][]float64
	// Backward pass, propagating gradients.
	Backward(dOutput [][]float64) [][]float64
	// Update learnable parameters.
	Update(learningRate float64)
}

// --- Faithful Dendritic Neuron Implementation ---

type DendriticCompartment struct {
	weights         []float64
	feedbackWeights []float64
	bias            float64
	grads           []float64
	biasGrad        float64 // Gradient for the bias term.
	lastInput       []float64
	lastOutput      float64
	inputIndices    []int
	inFeatures, numConnections int
}

func NewDendriticCompartment(inFeatures, numConnections int) *DendriticCompartment {
	if numConnections > inFeatures {
		numConnections = inFeatures
	}
	return &DendriticCompartment{
		weights:         newMatrix(numConnections, 1),
		feedbackWeights: newMatrix(numConnections, 1),
		bias:            rand.NormFloat64(),
		grads:           make([]float64, numConnections),
		inputIndices:    rand.Perm(inFeatures)[:numConnections],
		inFeatures:      inFeatures,
		numConnections:  numConnections,
	}
}

func (dc *DendriticCompartment) Forward(input []float64) float64 {
	dc.lastInput = make([]float64, dc.numConnections)
	sum := dc.bias
	for i := 0; i < dc.numConnections; i++ {
		idx := dc.inputIndices[i]
		dc.lastInput[i] = input[idx]
		sum += dc.lastInput[i] * dc.weights[i]
	}
	dc.lastOutput = math.Tanh(sum)
	return dc.lastOutput
}

func (dc *DendriticCompartment) Backward(dOutput float64) []float64 {
	// Backprop through Tanh to get the local error signal.
	dTanh := (1 - dc.lastOutput*dc.lastOutput) * dOutput

	// The gradient for the bias is the local error signal itself.
	dc.biasGrad = dTanh

	// Calculate gradients for weights using the local error.
	for i := 0; i < dc.numConnections; i++ {
		dc.grads[i] = dc.lastInput[i] * dTanh
	}

	// Propagate error backward to the input layer using fixed feedback weights.
	dInput := make([]float64, dc.inFeatures)
	for i := 0; i < dc.numConnections; i++ {
		idx := dc.inputIndices[i]
		dInput[idx] += dTanh * dc.feedbackWeights[i]
	}
	return dInput
}

func (dc *DendriticCompartment) Update(learningRate float64) {
	for i := range dc.weights {
		dc.weights[i] -= learningRate * dc.grads[i]
	}
	// Correctly update the bias using its own gradient.
	dc.bias -= learningRate * dc.biasGrad
}

type DendriticNeuron struct {
	compartments    []*DendriticCompartment
	somaWeights     []float64
	feedbackSoma    []float64 // Fixed random weights for FA
	somaBias        float64
	lastCompartsOut []float64
	somaGrads       []float64
	somaBiasGrad    float64
}

func NewDendriticNeuron(inFeatures, numCompartments, connsPerCompart int) *DendriticNeuron {
	compartments := make([]*DendriticCompartment, numCompartments)
	for i := 0; i < numCompartments; i++ {
		compartments[i] = NewDendriticCompartment(inFeatures, connsPerCompart)
	}

	return &DendriticNeuron{
		compartments:    compartments,
		somaWeights:     newMatrix(numCompartments, 1),
		feedbackSoma:    newMatrix(numCompartments, 1),
		somaBias:        rand.NormFloat64(),
		somaGrads:       make([]float64, numCompartments),
	}
}

// Forward pass for a single neuron, processing a single data point (vector).
func (dn *DendriticNeuron) Forward(input []float64) float64 {
	dn.lastCompartsOut = make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		dn.lastCompartsOut[i] = compartment.Forward(input)
	}

	somaVal := dn.somaBias
	for i, out := range dn.lastCompartsOut {
		somaVal += out * dn.somaWeights[i]
	}
	// In a real transformer, this would also have an activation function (e.g. ReLU)
	// and the backward pass would account for its derivative.
	return somaVal
}

func (dn *DendriticNeuron) Backward(dOutput float64) []float64 {
	// Step A: Calculate gradients for the soma.
	dn.somaBiasGrad = dOutput
	for i := 0; i < len(dn.somaWeights); i++ {
		dn.somaGrads[i] = dOutput * dn.lastCompartsOut[i]
	}

	// Step B, C, D: Propagate error to compartments using feedback weights and run their backward pass.
	total_dInput := make([]float64, dn.compartments[0].inFeatures)
	for i, compartment := range dn.compartments {
		// Use the fixed random feedback weights to propagate the error to the compartment.
		dCompartmentOut := dOutput * dn.feedbackSoma[i]
		dCompartmentInput := compartment.Backward(dCompartmentOut)

		// Step E: Aggregate the errors from all compartments.
		for j, val := range dCompartmentInput {
			total_dInput[j] += val
		}
	}

	return total_dInput
}

func (dn *DendriticNeuron) Update(learningRate float64) {
	// Update the soma's parameters.
	dn.somaBias -= learningRate * dn.somaBiasGrad
	for i := range dn.somaWeights {
		dn.somaWeights[i] -= learningRate * dn.somaGrads[i]
	}

	// Trigger updates for all compartments.
	for _, compartment := range dn.compartments {
		compartment.Update(learningRate)
	}
}

// --- Faithful dANN FFN Module ---
// This module orchestrates a layer of DendriticNeurons.

type DANN_FFN struct {
	neurons []*DendriticNeuron
}

func NewDANN_FFN(inFeatures, outFeatures, numCompartments, connsPerCompart int) *DANN_FFN {
	neurons := make([]*DendriticNeuron, outFeatures)
	for i := 0; i < outFeatures; i++ {
		neurons[i] = NewDendriticNeuron(inFeatures, numCompartments, connsPerCompart)
	}
	return &DANN_FFN{neurons: neurons}
}

func (ffn *DANN_FFN) Forward(input [][]float64) [][]float64 {
	// Input is a sequence of vectors (seqLen, inFeatures).
	seqLen := len(input)
	outFeatures := len(ffn.neurons)
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, outFeatures)
		// For each item in the sequence, run it through all neurons in the layer.
		for j, neuron := range ffn.neurons {
			output[i][j] = neuron.Forward(input[i])
		}
	}
	return output
}

func (ffn *DANN_FFN) Backward(dOutput [][]float64) [][]float64 {
	// dOutput is (seqLen, outFeatures).
	seqLen := len(dOutput)
	inFeatures := ffn.neurons[0].compartments[0].inFeatures
	dInput := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		dInput[i] = make([]float64, inFeatures)
		// For each item in the sequence, propagate the error back through all neurons.
		for j, neuron := range ffn.neurons {
			// The error for the j-th neuron's output at this position in the sequence.
			dNeuronOutput := dOutput[i][j]
			dNeuronInput := neuron.Backward(dNeuronOutput)
			// Aggregate the error signals for the input features.
			for k := 0; k < inFeatures; k++ {
				dInput[i][k] += dNeuronInput[k]
			}
		}
	}
	return dInput
}

func (ffn *DANN_FFN) Update(learningRate float64) {
	for _, neuron := range ffn.neurons {
		neuron.Update(learningRate)
	}
}

// --- Standard Transformer Components (Blueprint) ---

// --- Linear Layer (used by Attention) ---
type Linear struct {
	weights         []float64
	feedbackWeights []float64
	biases          []float64
	grads           []float64
	lastInput       [][]float64
	inFeatures, outFeatures int
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	return &Linear{
		weights:         newMatrix(inFeatures, outFeatures),
		feedbackWeights: newMatrix(inFeatures, outFeatures),
		biases:          make([]float64, outFeatures),
		grads:           make([]float64, inFeatures*outFeatures),
		inFeatures:      inFeatures,
		outFeatures:     outFeatures,
	}
}

func (l *Linear) Forward(input [][]float64) [][]float64 {
	l.lastInput = input
	output := make([][]float64, len(input))
	for i := 0; i < len(input); i++ {
		output[i] = make([]float64, l.outFeatures)
		for j := 0; j < l.outFeatures; j++ {
			sum := l.biases[j]
			for k := 0; k < l.inFeatures; k++ {
				sum += input[i][k] * l.weights[k*l.outFeatures+j]
			}
			output[i][j] = sum
		}
	}
	return output
}

func (l *Linear) Backward(dOutput [][]float64) [][]float64 {
	// Calculate weight gradients.
	for i := 0; i < l.inFeatures; i++ {
		for j := 0; j < l.outFeatures; j++ {
			grad := 0.0
			for k := 0; k < len(dOutput); k++ {
				grad += l.lastInput[k][i] * dOutput[k][j]
			}
			l.grads[i*l.outFeatures+j] = grad
		}
	}
	// Propagate error backward using feedback weights.
	dInput := make([][]float64, len(dOutput))
	for i := 0; i < len(dOutput); i++ {
		dInput[i] = make([]float64, l.inFeatures)
		for j := 0; j < l.inFeatures; j++ {
			sum := 0.0
			for k := 0; k < l.outFeatures; k++ {
				sum += dOutput[i][k] * l.feedbackWeights[j*l.outFeatures+k]
			}
			dInput[i][j] = sum
		}
	}
	return dInput
}

func (l *Linear) Update(learningRate float64) {
	for i := range l.weights {
		l.weights[i] -= learningRate * l.grads[i]
	}
	// Bias update would also be needed here in a full implementation.
}


// --- Layer Normalization Module ---
type LayerNorm struct {
	gamma, beta []float64
	lastInput   [][]float64
	mean, variance []float64
	dModel      int
}

func NewLayerNorm(dModel int) *LayerNorm {
	ln := &LayerNorm{
		gamma:  make([]float64, dModel),
		beta:   make([]float64, dModel),
		dModel: dModel,
	}
	for i := range ln.gamma {
		ln.gamma[i] = 1.0
	}
	return ln
}

func (ln *LayerNorm) Forward(input [][]float64) [][]float64 {
	ln.lastInput = input
	output := make([][]float64, len(input))
	ln.mean = make([]float64, len(input))
	ln.variance = make([]float64, len(input))
	epsilon := 1e-6

	for i, row := range input {
		mean := 0.0
		for _, val := range row {
			mean += val
		}
		mean /= float64(ln.dModel)
		ln.mean[i] = mean

		variance := 0.0
		for _, val := range row {
			variance += math.Pow(val-mean, 2)
		}
		variance /= float64(ln.dModel)
		ln.variance[i] = variance

		std := math.Sqrt(variance + epsilon)
		output[i] = make([]float64, ln.dModel)
		for j, val := range row {
			output[i][j] = (val-mean)/std*ln.gamma[j] + ln.beta[j]
		}
	}
	return output
}

func (ln *LayerNorm) Backward(dOutput [][]float64) [][]float64 {
	// BLUEPRINT: Full backward pass for LayerNorm is complex.
	// For now, we pass the gradient through as a placeholder.
	return dOutput
}

func (ln *LayerNorm) Update(learningRate float64) {
	// BLUEPRINT: Gamma and Beta are learnable, but their gradients are complex.
	// This would be implemented after the full backward pass is complete.
}

// --- Add & Norm Module (Residual Connection + LayerNorm) ---
type AddAndNorm struct {
	layerNorm *LayerNorm
}

func NewAddAndNorm(dModel int) *AddAndNorm {
	return &AddAndNorm{
		layerNorm: NewLayerNorm(dModel),
	}
}

func (an *AddAndNorm) Forward(input, residual [][]float64) [][]float64 {
	added := make([][]float64, len(input))
	for i := range input {
		added[i] = make([]float64, len(input[i]))
		for j := range input[i] {
			added[i][j] = input[i][j] + residual[i][j]
		}
	}
	return an.layerNorm.Forward(added)
}

func (an *AddAndNorm) Backward(dOutput [][]float64) ([][]float64, [][]float64) {
	// Backprop through LayerNorm.
	dLayerNormInput := an.layerNorm.Backward(dOutput)
	// Gradients for both the main input and the residual are the same.
	return dLayerNormInput, dLayerNormInput
}

func (an *AddAndNorm) Update(learningRate float64) {
	an.layerNorm.Update(learningRate)
}

// --- Multi-Head Attention Module ---
type MultiHeadAttention struct {
	q_proj, k_proj, v_proj *Linear
	out_proj               *Linear
	numHeads               int
	dModel, headDim        int
}

func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
	headDim := dModel / numHeads
	return &MultiHeadAttention{
		q_proj:   NewLinear(dModel, dModel),
		k_proj:   NewLinear(dModel, dModel),
		v_proj:   NewLinear(dModel, dModel),
		out_proj: NewLinear(dModel, dModel),
		numHeads: numHeads,
		dModel:   dModel,
		headDim:  headDim,
	}
}

func (mha *MultiHeadAttention) Forward(query, key, value [][]float64) [][]float64 {
	seqLen := len(query)
	q := mha.q_proj.Forward(query)
	k := mha.k_proj.Forward(key)
	v := mha.v_proj.Forward(value)

	attentionOutput := make([][]float64, seqLen)
	for i := range attentionOutput {
		attentionOutput[i] = make([]float64, mha.dModel)
	}

	for h := 0; h < mha.numHeads; h++ {
		q_h := make([][]float64, seqLen)
		k_h := make([][]float64, seqLen)
		v_h := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			q_h[i] = q[i][h*mha.headDim : (h+1)*mha.headDim]
			k_h[i] = k[i][h*mha.headDim : (h+1)*mha.headDim]
			v_h[i] = v[i][h*mha.headDim : (h+1)*mha.headDim]
		}

		scores := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			scores[i] = make([]float64, seqLen)
			for j := 0; j < seqLen; j++ {
				for d := 0; d < mha.headDim; d++ {
					scores[i][j] += q_h[i][d] * k_h[j][d]
				}
				scores[i][j] /= math.Sqrt(float64(mha.headDim))
			}
		}

		// Softmax scores would be here...

		// Apply scores to Value would be here...

		// Concatenate head outputs would be here...
	}

	return mha.out_proj.Forward(attentionOutput)
}

func (mha *MultiHeadAttention) Backward(dOutput [][]float64) [][]float64 {
	// BLUEPRINT: Full backward pass for MHA is very complex.
	dAttentionOutput := mha.out_proj.Backward(dOutput)
	// This would be followed by backprop through concatenation, value, scores, and projections.
	return dAttentionOutput
}

func (mha *MultiHeadAttention) Update(learningRate float64) {
	mha.q_proj.Update(learningRate)
	mha.k_proj.Update(learningRate)
	mha.v_proj.Update(learningRate)
	mha.out_proj.Update(learningRate)
}

// --- Full Transformer Block ---
// This struct shows how the components are wired together.

type TransformerBlock struct {
	attention   *MultiHeadAttention
	addNorm1    *AddAndNorm
	ffn         *DANN_FFN // The dANN Feed-Forward Network
	addNorm2    *AddAndNorm
	dModel      int
}

func NewTransformerBlock(dModel, numHeads, numCompartments, connsPerCompart int) *TransformerBlock {
	return &TransformerBlock{
		attention:   NewMultiHeadAttention(dModel, numHeads),
		addNorm1:    NewAddAndNorm(dModel),
		ffn:         NewDANN_FFN(dModel, dModel, numCompartments, connsPerCompart),
		addNorm2:    NewAddAndNorm(dModel),
		dModel:      dModel,
	}
}

func (tb *TransformerBlock) Forward(input [][]float64) [][]float64 {
	// 1. Multi-Head Attention with residual connection and LayerNorm
	attentionOutput := tb.attention.Forward(input, input, input)
	addNorm1Output := tb.addNorm1.Forward(attentionOutput, input)

	// 2. dANN Feed-Forward Network with residual connection and LayerNorm
	ffnOutput := tb.ffn.Forward(addNorm1Output)
	addNorm2Output := tb.addNorm2.Forward(ffnOutput, addNorm1Output)

	return addNorm2Output
}

func (tb *TransformerBlock) Backward(dOutput [][]float64) [][]float64 {
	// BLUEPRINT: This shows the flow of gradients back through the block.
	// 1. Backprop through the second Add & Norm
	dFFNOutput, dAddNorm1Output_residual := tb.addNorm2.Backward(dOutput)

	// 2. Backprop through the dANN FFN
	dAddNorm1Output_main := tb.ffn.Backward(dFFNOutput)

	// 3. Combine gradients from the main path and the residual connection
	dAddNorm1Output := make([][]float64, len(dAddNorm1Output_main))
	// ... combination logic would be here ...

	// 4. Backprop through the first Add & Norm
	dAttentionOutput, dInput_residual := tb.addNorm1.Backward(dAddNorm1Output)

	// 5. Backprop through the Multi-Head Attention
	dInput_main := tb.attention.Backward(dAttentionOutput)

	// 6. Combine gradients from the main path and the residual connection
	dInput := make([][]float64, len(dInput_main))
	// ... combination logic would be here ...

	return dInput
}

func (tb *TransformerBlock) Update(learningRate float64) {
	tb.attention.Update(learningRate)
	tb.addNorm1.Update(learningRate)
	tb.ffn.Update(learningRate)
	tb.addNorm2.Update(learningRate)
}


func main() {
	fmt.Println("This file is a blueprint for the faithful implementation of a dANN-Transformer.")
	fmt.Println("It contains corrected and de-duplicated components, ready for further implementation.")
}
