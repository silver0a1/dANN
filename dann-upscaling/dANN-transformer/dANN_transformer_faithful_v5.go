package main

import (
	"fmt"
	"math"
	"math/rand"
)

// --- Utility Functions ---

func newMatrix(rows, cols int) []float64 {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(1.0/float64(cols))
	}
	return data
}

func transpose(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}
	return transposed
}

func matMul(a, b [][]float64) [][]float64 {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if colsA != rowsB {
		panic("Matrix dimensions are not compatible for multiplication")
	}
	result := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		result[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// --- Faithful Dendritic Neuron Implementation (from v2) ---

type DendriticCompartment struct {
	weights         []float64
	feedbackWeights []float64
	bias            float64
	grads           []float64
	biasGrad        float64
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
	dTanh := (1 - dc.lastOutput*dc.lastOutput) * dOutput
	dc.biasGrad = dTanh
	for i := 0; i < dc.numConnections; i++ {
		dc.grads[i] = dc.lastInput[i] * dTanh
	}
	dInput := make([]float64, dc.inFeatures)
	for i := 0; i < dc.numConnections; i++ {
		idx := dc.inputIndices[i]
		dInput[idx] += dTanh * dc.feedbackWeights[i]
	}
	return dInput
}

func (dc *DendriticCompartment) Update(learningRate float64) {
    clipValue := 1.0
	for i := range dc.weights {
        if dc.grads[i] > clipValue {
            dc.grads[i] = clipValue
        }
        if dc.grads[i] < -clipValue {
            dc.grads[i] = -clipValue
        }
		dc.weights[i] -= learningRate * dc.grads[i]
	}
    if dc.biasGrad > clipValue {
        dc.biasGrad = clipValue
    }
    if dc.biasGrad < -clipValue {
        dc.biasGrad = -clipValue
    }
	dc.bias -= learningRate * dc.biasGrad
}

type DendriticNeuron struct {
	compartments    []*DendriticCompartment
	somaWeights     []float64
	feedbackSoma    []float64
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

func (dn *DendriticNeuron) Forward(input []float64) float64 {
	dn.lastCompartsOut = make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		dn.lastCompartsOut[i] = compartment.Forward(input)
	}
	somaVal := dn.somaBias
	for i, out := range dn.lastCompartsOut {
		somaVal += out * dn.somaWeights[i]
	}
	return somaVal
}

func (dn *DendriticNeuron) Backward(dOutput float64) []float64 {
	dn.somaBiasGrad = dOutput
	for i := 0; i < len(dn.somaWeights); i++ {
		dn.somaGrads[i] = dOutput * dn.lastCompartsOut[i]
	}
	total_dInput := make([]float64, dn.compartments[0].inFeatures)
	for i, compartment := range dn.compartments {
		dCompartmentOut := dOutput * dn.feedbackSoma[i]
		dCompartmentInput := compartment.Backward(dCompartmentOut)
		for j, val := range dCompartmentInput {
			total_dInput[j] += val
		}
	}
	return total_dInput
}

func (dn *DendriticNeuron) Update(learningRate float64) {
    clipValue := 1.0
	dn.somaBias -= learningRate * dn.somaBiasGrad
	for i := range dn.somaWeights {
        if dn.somaGrads[i] > clipValue {
            dn.somaGrads[i] = clipValue
        }
        if dn.somaGrads[i] < -clipValue {
            dn.somaGrads[i] = -clipValue
        }
		dn.somaWeights[i] -= learningRate * dn.somaGrads[i]
	}
	for _, compartment := range dn.compartments {
		compartment.Update(learningRate)
	}
}

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
	seqLen := len(input)
	outFeatures := len(ffn.neurons)
	output := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, outFeatures)
		for j, neuron := range ffn.neurons {
			output[i][j] = neuron.Forward(input[i])
		}
	}
	return output
}

func (ffn *DANN_FFN) Backward(dOutput [][]float64) [][]float64 {
	seqLen := len(dOutput)
	inFeatures := ffn.neurons[0].compartments[0].inFeatures
	dInput := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		dInput[i] = make([]float64, inFeatures)
		for j, neuron := range ffn.neurons {
			dNeuronOutput := dOutput[i][j]
			dNeuronInput := neuron.Backward(dNeuronOutput)
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

// --- Standard Transformer Components (with Backpropagation) ---

// --- Linear Layer ---
type Linear struct {
	weights         []float64
	biases          []float64
	weightGrads     []float64
	biasGrads       []float64
	lastInput       [][]float64
	inFeatures, outFeatures int
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	return &Linear{
		weights:         newMatrix(inFeatures, outFeatures),
		biases:          make([]float64, outFeatures),
		weightGrads:     make([]float64, inFeatures*outFeatures),
		biasGrads:       make([]float64, outFeatures),
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
	for i := range l.weightGrads {
		l.weightGrads[i] = 0
	}
	for i := range l.biasGrads {
		l.biasGrads[i] = 0
	}
	dInput := make([][]float64, len(l.lastInput))
	for i := 0; i < len(l.lastInput); i++ {
		dInput[i] = make([]float64, l.inFeatures)
		for j := 0; j < l.outFeatures; j++ {
			l.biasGrads[j] += dOutput[i][j]
			for k := 0; k < l.inFeatures; k++ {
				l.weightGrads[k*l.outFeatures+j] += l.lastInput[i][k] * dOutput[i][j]
				dInput[i][k] += l.weights[k*l.outFeatures+j] * dOutput[i][j]
			}
		}
	}
	return dInput
}

func (l *Linear) Update(learningRate float64) {
    // Add gradient clipping
    clipValue := 1.0
	for i := range l.weights {
        if l.weightGrads[i] > clipValue {
            l.weightGrads[i] = clipValue
        }
        if l.weightGrads[i] < -clipValue {
            l.weightGrads[i] = -clipValue
        }
		l.weights[i] -= learningRate * l.weightGrads[i]
	}
	for i := range l.biases {
        if l.biasGrads[i] > clipValue {
            l.biasGrads[i] = clipValue
        }
        if l.biasGrads[i] < -clipValue {
            l.biasGrads[i] = -clipValue
        }
		l.biases[i] -= learningRate * l.biasGrads[i]
	}
}

// --- Layer Normalization ---
type LayerNorm struct {
	gamma, beta []float64
	gammaGrads, betaGrads []float64
	lastInput, normalizedInput [][]float64
	mean, variance []float64
	dModel int
	epsilon float64
}

func NewLayerNorm(dModel int) *LayerNorm {
	ln := &LayerNorm{
		gamma:      make([]float64, dModel),
		beta:       make([]float64, dModel),
		gammaGrads: make([]float64, dModel),
		betaGrads:  make([]float64, dModel),
		dModel:     dModel,
		epsilon:    1e-6,
	}
	for i := range ln.gamma {
		ln.gamma[i] = 1.0
	}
	return ln
}

func (ln *LayerNorm) Forward(input [][]float64) [][]float64 {
	ln.lastInput = input
	seqLen := len(input)
	ln.normalizedInput = make([][]float64, seqLen)
	ln.mean = make([]float64, seqLen)
	ln.variance = make([]float64, seqLen)
	output := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		mean := 0.0
		for _, val := range input[i] {
			mean += val
		}
		mean /= float64(ln.dModel)
		ln.mean[i] = mean
		variance := 0.0
		for _, val := range input[i] {
			variance += math.Pow(val - mean, 2)
		}
		variance /= float64(ln.dModel)
		ln.variance[i] = variance
		std_inv := 1.0 / math.Sqrt(variance+ln.epsilon)
		ln.normalizedInput[i] = make([]float64, ln.dModel)
		output[i] = make([]float64, ln.dModel)
		for j := 0; j < ln.dModel; j++ {
			ln.normalizedInput[i][j] = (input[i][j] - mean) * std_inv
			output[i][j] = ln.normalizedInput[i][j]*ln.gamma[j] + ln.beta[j]
		}
	}
	return output
}

func (ln *LayerNorm) Backward(dOutput [][]float64) [][]float64 {
	seqLen := len(dOutput)
	dInput := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		dInput[i] = make([]float64, ln.dModel)
		std_inv := 1.0 / math.Sqrt(ln.variance[i]+ln.epsilon)
		dNorm := make([]float64, ln.dModel)
		for j := 0; j < ln.dModel; j++ {
			ln.betaGrads[j] += dOutput[i][j]
			ln.gammaGrads[j] += dOutput[i][j] * ln.normalizedInput[i][j]
			dNorm[j] = dOutput[i][j] * ln.gamma[j]
		}
		dVariance := 0.0
		for j := 0; j < ln.dModel; j++ {
			dVariance += dNorm[j] * (ln.lastInput[i][j] - ln.mean[i]) * (-0.5) * math.Pow(std_inv, 3)
		}
		dMean := 0.0
		for j := 0; j < ln.dModel; j++ {
			dMean += -dNorm[j] * std_inv
			dMean += -2.0 * dVariance * (ln.lastInput[i][j] - ln.mean[i]) / float64(ln.dModel)
		}
		for j := 0; j < ln.dModel; j++ {
			dInput[i][j] = dNorm[j]*std_inv + dVariance*2.0*(ln.lastInput[i][j]-ln.mean[i])/float64(ln.dModel) + dMean/float64(ln.dModel)
		}
	}
	return dInput
}

func (ln *LayerNorm) Update(learningRate float64) {
    clipValue := 1.0
	for i := range ln.gamma {
        if ln.gammaGrads[i] > clipValue {
            ln.gammaGrads[i] = clipValue
        }
        if ln.gammaGrads[i] < -clipValue {
            ln.gammaGrads[i] = -clipValue
        }
		ln.gamma[i] -= learningRate * ln.gammaGrads[i]

        if ln.betaGrads[i] > clipValue {
            ln.betaGrads[i] = clipValue
        }
        if ln.betaGrads[i] < -clipValue {
            ln.betaGrads[i] = -clipValue
        }
		ln.beta[i] -= learningRate * ln.betaGrads[i]
	}
	for i := range ln.gammaGrads {
		ln.gammaGrads[i] = 0
		ln.betaGrads[i] = 0
	}
}

// --- Add & Norm ---
type AddAndNorm struct {
	layerNorm *LayerNorm
}

func NewAddAndNorm(dModel int) *AddAndNorm {
	return &AddAndNorm{layerNorm: NewLayerNorm(dModel)}
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
	dLayerNormInput := an.layerNorm.Backward(dOutput)
	return dLayerNormInput, dLayerNormInput
}

func (an *AddAndNorm) Update(learningRate float64) {
	an.layerNorm.Update(learningRate)
}

// --- Multi-Head Attention ---
type MultiHeadAttention struct {
	q_proj, k_proj, v_proj *Linear
	out_proj               *Linear
	numHeads               int
	dModel, headDim        int
	lastQ, lastK, lastV    [][][]float64 // Per head: [head][seqLen][headDim]
	lastScores             [][][]float64 // Per head: [head][seqLen][seqLen]
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
	q_full := mha.q_proj.Forward(query)
	k_full := mha.k_proj.Forward(key)
	v_full := mha.v_proj.Forward(value)

	mha.lastQ = make([][][]float64, mha.numHeads)
	mha.lastK = make([][][]float64, mha.numHeads)
	mha.lastV = make([][][]float64, mha.numHeads)
	mha.lastScores = make([][][]float64, mha.numHeads)

	attentionOutputConcat := make([][]float64, seqLen)
	for i := range attentionOutputConcat {
		attentionOutputConcat[i] = make([]float64, mha.dModel)
	}

	for h := 0; h < mha.numHeads; h++ {
		q_h := make([][]float64, seqLen)
		k_h := make([][]float64, seqLen)
		v_h := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			q_h[i] = q_full[i][h*mha.headDim : (h+1)*mha.headDim]
			k_h[i] = k_full[i][h*mha.headDim : (h+1)*mha.headDim]
			v_h[i] = v_full[i][h*mha.headDim : (h+1)*mha.headDim]
		}
		mha.lastQ[h], mha.lastK[h], mha.lastV[h] = q_h, k_h, v_h

		scores := matMul(q_h, transpose(k_h))
		for i := range scores {
			for j := range scores[i] {
				scores[i][j] /= math.Sqrt(float64(mha.headDim))
			}
		}

		// Softmax
		for i := 0; i < seqLen; i++ {
			maxVal := scores[i][0]
			for _, s := range scores[i] {
				if s > maxVal {
					maxVal = s
				}
			}
			sumExp := 0.0
			for j, s := range scores[i] {
				scores[i][j] = math.Exp(s - maxVal)
				sumExp += scores[i][j]
			}
			for j := range scores[i] {
				scores[i][j] /= sumExp
			}
		}
		mha.lastScores[h] = scores

		headOutput := matMul(scores, v_h)
		for i := 0; i < seqLen; i++ {
			copy(attentionOutputConcat[i][h*mha.headDim:], headOutput[i])
		}
	}

	return mha.out_proj.Forward(attentionOutputConcat)
}

func (mha *MultiHeadAttention) Backward(dOutput [][]float64) ([][]float64, [][]float64, [][]float64) {
	dAttentionOutputConcat := mha.out_proj.Backward(dOutput)
	seqLen := len(dOutput)

	dQ_full := make([][]float64, seqLen)
	dK_full := make([][]float64, seqLen)
	dV_full := make([][]float64, seqLen)
	for i := range dQ_full {
		dQ_full[i] = make([]float64, mha.dModel)
		dK_full[i] = make([]float64, mha.dModel)
		dV_full[i] = make([]float64, mha.dModel)
	}

	for h := 0; h < mha.numHeads; h++ {
		dHeadOutput := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			dHeadOutput[i] = dAttentionOutputConcat[i][h*mha.headDim : (h+1)*mha.headDim]
		}

		dScores := matMul(dHeadOutput, transpose(mha.lastV[h]))
		dV_h := matMul(transpose(mha.lastScores[h]), dHeadOutput)

		// Backward through Softmax
		for i := 0; i < seqLen; i++ {
			row := mha.lastScores[h][i]
			dRow := dScores[i]
			dot := 0.0
			for j := 0; j < seqLen; j++ {
				dot += dRow[j] * row[j]
			}
			for j := 0; j < seqLen; j++ {
				dScores[i][j] = row[j] * (dRow[j] - dot)
			}
		}

		scale := 1.0 / math.Sqrt(float64(mha.headDim))
		for i := range dScores {
			for j := range dScores[i] {
				dScores[i][j] *= scale
			}
		}

		dQ_h := matMul(dScores, mha.lastK[h])
		dK_h := matMul(transpose(dScores), mha.lastQ[h])

		for i := 0; i < seqLen; i++ {
			copy(dQ_full[i][h*mha.headDim:], dQ_h[i])
			copy(dK_full[i][h*mha.headDim:], dK_h[i])
			copy(dV_full[i][h*mha.headDim:], dV_h[i])
		}
	}

	dQuery := mha.q_proj.Backward(dQ_full)
	dKey := mha.k_proj.Backward(dK_full)
	dValue := mha.v_proj.Backward(dV_full)

	return dQuery, dKey, dValue
}

func (mha *MultiHeadAttention) Update(learningRate float64) {
	mha.q_proj.Update(learningRate)
	mha.k_proj.Update(learningRate)
	mha.v_proj.Update(learningRate)
	mha.out_proj.Update(learningRate)
}

// --- Full Transformer Block ---
type TransformerBlock struct {
	attention   *MultiHeadAttention
	addNorm1    *AddAndNorm
	ffn         *DANN_FFN
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
	attentionOutput := tb.attention.Forward(input, input, input)
	addNorm1Output := tb.addNorm1.Forward(attentionOutput, input)
	ffnOutput := tb.ffn.Forward(addNorm1Output)
	addNorm2Output := tb.addNorm2.Forward(ffnOutput, addNorm1Output)
	return addNorm2Output
}

func (tb *TransformerBlock) Backward(dOutput [][]float64) [][]float64 {
	dFFNOutput, dAddNorm1Output_residual := tb.addNorm2.Backward(dOutput)
	dAddNorm1Output_main := tb.ffn.Backward(dFFNOutput)
	dAddNorm1Output := make([][]float64, len(dAddNorm1Output_main))
	for i := range dAddNorm1Output {
		dAddNorm1Output[i] = make([]float64, len(dAddNorm1Output_main[0]))
		for j := range dAddNorm1Output[i] {
			dAddNorm1Output[i][j] = dAddNorm1Output_main[i][j] + dAddNorm1Output_residual[i][j]
		}
	}
	dAttentionOutput, dInput_residual := tb.addNorm1.Backward(dAddNorm1Output)
	dQuery, dKey, dValue := tb.attention.Backward(dAttentionOutput)
	dInput := make([][]float64, len(dQuery))
	for i := range dInput {
		dInput[i] = make([]float64, len(dQuery[0]))
		for j := range dInput[i] {
			dInput[i][j] = dQuery[i][j] + dKey[i][j] + dValue[i][j] + dInput_residual[i][j]
		}
	}
	return dInput
}

func (tb *TransformerBlock) Update(learningRate float64) {
	tb.attention.Update(learningRate)
	tb.addNorm1.Update(learningRate)
	tb.ffn.Update(learningRate)
	tb.addNorm2.Update(learningRate)
}

// --- Full DANN-Transformer Model ---
type DANNTransformer struct {
	tokenEmbedding    *Linear
	positionalEncoding [][]float64
	blocks            []*TransformerBlock
	outputLayer       *Linear
	vocabSize         int
	dModel            int
	seqLen            int
}

func NewDANNTransformer(vocabSize, dModel, seqLen, numBlocks, numHeads, numCompartments, connsPerCompart int) *DANNTransformer {
	blocks := make([]*TransformerBlock, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = NewTransformerBlock(dModel, numHeads, numCompartments, connsPerCompart)
	}
	return &DANNTransformer{
		tokenEmbedding:    NewLinear(vocabSize, dModel),
		positionalEncoding: createPositionalEncoding(seqLen, dModel),
		blocks:            blocks,
		outputLayer:       NewLinear(dModel, vocabSize),
		vocabSize:         vocabSize,
		dModel:            dModel,
		seqLen:            seqLen,
	}
}

func createPositionalEncoding(seqLen, dModel int) [][]float64 {
	pe := make([][]float64, seqLen)
	for i := range pe {
		pe[i] = make([]float64, dModel)
	}
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel; i++ {
			if i%2 == 0 {
				pe[pos][i] = math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(dModel)))
			} else {
				pe[pos][i] = math.Cos(float64(pos) / math.Pow(10000, float64(i-1)/float64(dModel)))
			}
		}
	}
	return pe
}

func (t *DANNTransformer) Forward(input []int) [][]float64 {
	// 1. Token Embedding
	embedded := t.tokenEmbedding.Forward(oneHotEncode(input, t.vocabSize))

	// 2. Add Positional Encoding
	for i := 0; i < t.seqLen; i++ {
		for j := 0; j < t.dModel; j++ {
			embedded[i][j] += t.positionalEncoding[i][j]
		}
	}

	// 3. Transformer Blocks
	blockInput := embedded
	for _, block := range t.blocks {
		blockInput = block.Forward(blockInput)
	}

	// 4. Output Layer
	return t.outputLayer.Forward(blockInput)
}

func (t *DANNTransformer) Backward(dOutput [][]float64) {
	// 1. Backward through Output Layer
	dBlockOutput := t.outputLayer.Backward(dOutput)

	// 2. Backward through Transformer Blocks
	for i := len(t.blocks) - 1; i >= 0; i-- {
		dBlockOutput = t.blocks[i].Backward(dBlockOutput)
	}

	// 3. Backward through Positional Encoding (gradient is passed through)
	// 4. Backward through Token Embedding
	t.tokenEmbedding.Backward(dBlockOutput)
}

func (t *DANNTransformer) Update(learningRate float64) {
	t.tokenEmbedding.Update(learningRate)
	for _, block := range t.blocks {
		block.Update(learningRate)
	}
	t.outputLayer.Update(learningRate)
}

func oneHotEncode(input []int, vocabSize int) [][]float64 {
	encoded := make([][]float64, len(input))
	for i, val := range input {
		encoded[i] = make([]float64, vocabSize)
		encoded[i][val] = 1.0
	}
	return encoded
}

// --- Gradient Checking ---

func calculateLoss(model *DANNTransformer, inputSeq, targetSeq []int) float64 {
	outputLogits := model.Forward(inputSeq)
	loss := 0.0
	for i := 0; i < len(inputSeq); i++ {
		targetIdx := targetSeq[i]
		logits := outputLogits[i]
		maxLogit := logits[0]
		for _, l := range logits {
			if l > maxLogit {
				maxLogit = l
			}
		}
		sumExp := 0.0
		probs := make([]float64, len(logits))
		for j, l := range logits {
			probs[j] = math.Exp(l - maxLogit)
			sumExp += probs[j]
		}
		for j := range probs {
			probs[j] /= sumExp
		}
		loss -= math.Log(probs[targetIdx])
	}
	return loss / float64(len(inputSeq))
}

func main() {
	// Hyperparameters
	vocabSize := 10
	dModel := 16 // Smaller model for faster checking
	seqLen := 4
	numBlocks := 1
	numHeads := 2
	numCompartments := 2
	connsPerCompart := 4

	// Create the model
	model := NewDANNTransformer(vocabSize, dModel, seqLen, numBlocks, numHeads, numCompartments, connsPerCompart)

	// Create dummy data
	inputSeq := make([]int, seqLen)
	targetSeq := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		val := rand.Intn(vocabSize-2) + 1
		inputSeq[i] = val
		targetSeq[seqLen-1-i] = val
	}

	// --- Analytical Gradient Calculation ---
	outputLogits := model.Forward(inputSeq)
	dLogits := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		dLogits[i] = make([]float64, vocabSize)
		targetIdx := targetSeq[i]
		logits := outputLogits[i]
		maxLogit := logits[0]
		for _, l := range logits {
			if l > maxLogit {
				maxLogit = l
			}
		}
		sumExp := 0.0
		probs := make([]float64, vocabSize)
		for j, l := range logits {
			probs[j] = math.Exp(l - maxLogit)
			sumExp += probs[j]
		}
		for j := range probs {
			probs[j] /= sumExp
		}
		for j := 0; j < vocabSize; j++ {
			if j == targetIdx {
				dLogits[i][j] = probs[j] - 1
			} else {
				dLogits[i][j] = probs[j]
			}
		}
	}
	model.Backward(dLogits)

	// --- Numerical Gradient Calculation ---
	epsilon := 1e-5
	// Check a single weight in the first linear layer of the first attention block
	param := &model.blocks[0].attention.q_proj.weights[0]
	analyticalGrad := model.blocks[0].attention.q_proj.weightGrads[0]

	// Calculate loss1
	*param += epsilon
	loss1 := calculateLoss(model, inputSeq, targetSeq)

	// Calculate loss2
	*param -= 2 * epsilon
	loss2 := calculateLoss(model, inputSeq, targetSeq)

	// Reset param
	*param += epsilon

	numericalGrad := (loss1 - loss2) / (2 * epsilon)

	fmt.Println("--- Gradient Check ---")
	fmt.Printf("Analytical Gradient: %f\n", analyticalGrad)
	fmt.Printf("Numerical Gradient:  %f\n", numericalGrad)

	relError := math.Abs(analyticalGrad-numericalGrad) / math.Max(math.Abs(analyticalGrad), math.Abs(numericalGrad))
	fmt.Printf("Relative Error:    %e\n", relError)

	if relError < 1e-5 {
		fmt.Println("Gradient check passed!")
	} else {
		fmt.Println("Gradient check FAILED!")
	}
}
