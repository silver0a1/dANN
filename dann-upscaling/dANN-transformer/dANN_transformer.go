package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- Core dANN Component (from dANN_tumor_bio.go) ---
// This will be used as the Feed-Forward Network (FFN) in the Transformer block.

type DendriticCompartment struct {
	weights      []float64
	bias         float64
	inputIndices []int
}

func NewDendriticCompartment(numTotalInputs, numConnections int) *DendriticCompartment {
	if numConnections > numTotalInputs {
		numConnections = numTotalInputs
	}
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
	for i, weight := range dc.weights {
		inputIndex := dc.inputIndices[i]
		sum += allInputs[inputIndex] * weight
	}
	return math.Tanh(sum)
}

type DendriticNeuron struct {
	compartments         []*DendriticCompartment
	somaWeights          []float64
	somaBias             float64
}

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
	compartmentOutputs := make([]float64, len(dn.compartments))
	for i, compartment := range dn.compartments {
		compartmentOutputs[i] = compartment.Process(inputs)
	}
	finalSum := dn.somaBias
	for i, output := range compartmentOutputs {
		finalSum += output * dn.somaWeights[i]
	}
	return finalSum
}

// This is the wrapper that makes our dANN a compatible FFN module.
type DANN_FFN struct {
	layer []*DendriticNeuron
	// In a real scenario, this would be more complex, maybe multiple layers.
}

func NewDANN_FFN(inputDim, outputDim int) *DANN_FFN {
	// For simplicity, we ensure the dANN's output matches the required dimension.
	// The internal structure (compartments, etc.) is a hyperparameter.
	layer := make([]*DendriticNeuron, outputDim)
	for i := 0; i < outputDim; i++ {
		// Example hyperparameters for the dendritic structure
		layer[i] = NewDendriticNeuron(inputDim, 8, 16)
	}
	return &DANN_FFN{layer: layer}
}

func (ffn *DANN_FFN) Forward(input []float64) []float64 {
	output := make([]float64, len(ffn.layer))
	for i, neuron := range ffn.layer {
		output[i] = neuron.Forward(input)
	}
	return output
}

// --- NEW: Transformer Components ---

// A simple matrix for holding weights
type Matrix struct {
	rows, cols int
	data       []float64
}

func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
	// Initialize with small random values
	for i := range m.data {
		m.data[i] = rand.NormFloat64() * math.Sqrt(2.0/float64(cols))
	}
	return m
}

// Simple linear layer
type Linear struct {
	weights *Matrix
	bias    []float64
}

func NewLinear(inputDim, outputDim int) *Linear {
	return &Linear{
		weights: NewMatrix(inputDim, outputDim),
		bias:    make([]float64, outputDim),
	}
}

func (l *Linear) Forward(input [][]float64) [][]float64 {
	// input is (batch_size, seq_len, input_dim)
	// for simplicity, we process one sequence at a time
	// input is (seq_len, input_dim)
	seqLen := len(input)
	out := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		out[i] = make([]float64, l.weights.cols)
		for j := 0; j < l.weights.cols; j++ {
			sum := l.bias[j]
			for k := 0; k < l.weights.rows; k++ {
				sum += input[i][k] * l.weights.data[k*l.weights.cols+j]
			}
			out[i][j] = sum
		}
	}
	return out
}

// Positional Encoding
func getPositionalEncoding(seqLen, dModel int) [][]float64 {
	pe := make([][]float64, seqLen)
	for i := range pe {
		pe[i] = make([]float64, dModel)
	}

	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel/2; i++ {
			divTerm := math.Pow(10000, float64(2*i)/float64(dModel))
			pe[pos][2*i] = math.Sin(float64(pos) / divTerm)
			pe[pos][2*i+1] = math.Cos(float64(pos) / divTerm)
		}
	}
	return pe
}

// Layer Normalization
func layerNorm(input [][]float64) [][]float64 {
	out := make([][]float64, len(input))
	epsilon := 1e-6

	for i, row := range input {
		mean := 0.0
		for _, val := range row {
			mean += val
		}
		mean /= float64(len(row))

		variance := 0.0
		for _, val := range row {
			variance += math.Pow(val-mean, 2)
		}
		variance /= float64(len(row))

		std := math.Sqrt(variance + epsilon)
		out[i] = make([]float64, len(row))
		for j, val := range row {
			out[i][j] = (val - mean) / std
		}
	}
	return out
}

// Simplified Self-Attention (Single Head for clarity)
func selfAttention(input [][]float64, q_proj, k_proj, v_proj *Linear) [][]float64 {
	seqLen := len(input)
	dModel := len(input[0])

	q := q_proj.Forward(input)
	k := k_proj.Forward(input)
	v := v_proj.Forward(input)

	// MatMul(q, k_transpose)
	scores := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		scores[i] = make([]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			for d := 0; d < dModel; d++ {
				scores[i][j] += q[i][d] * k[j][d]
			}
			scores[i][j] /= math.Sqrt(float64(dModel))
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

	// MatMul(scores, v)
	out := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		out[i] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			for k := 0; k < seqLen; k++ {
				out[i][j] += scores[i][k] * v[k][j]
			}
		}
	}
	return out
}

// Transformer Block
type TransformerBlock struct {
	q_proj, k_proj, v_proj *Linear
	dann_ffn               *DANN_FFN
}

func NewTransformerBlock(dModel int) *TransformerBlock {
	return &TransformerBlock{
		q_proj:   NewLinear(dModel, dModel),
		k_proj:   NewLinear(dModel, dModel),
		v_proj:   NewLinear(dModel, dModel),
		dann_ffn: NewDANN_FFN(dModel, dModel),
	}
}

func (b *TransformerBlock) Forward(input [][]float64) [][]float64 {
	// 1. Self-Attention
	attention_output := selfAttention(input, b.q_proj, b.k_proj, b.v_proj)

	// 2. Add & Norm
	add_norm_1 := make([][]float64, len(input))
	for i := range input {
		add_norm_1[i] = make([]float64, len(input[i]))
		for j := range input[i] {
			add_norm_1[i][j] = input[i][j] + attention_output[i][j]
		}
	}
	add_norm_1 = layerNorm(add_norm_1)

	// 3. dANN Feed-Forward Network
	ffn_output := make([][]float64, len(add_norm_1))
	for i, row := range add_norm_1 {
		ffn_output[i] = b.dann_ffn.Forward(row)
	}

	// 4. Add & Norm
	add_norm_2 := make([][]float64, len(add_norm_1))
	for i := range add_norm_1 {
		add_norm_2[i] = make([]float64, len(add_norm_1[i]))
		for j := range add_norm_1[i] {
			add_norm_2[i][j] = add_norm_1[i][j] + ffn_output[i][j]
		}
	}
	add_norm_2 = layerNorm(add_norm_2)

	return add_norm_2
}

// The Main dANN-Transformer Model
type DANNTransformer struct {
	embedding      *Linear
	pos_encoding   [][]float64
	blocks         []*TransformerBlock
	output_layer   *Linear
}

func NewDANNTransformer(vocabSize, seqLen, dModel, numBlocks int) *DANNTransformer {
	blocks := make([]*TransformerBlock, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = NewTransformerBlock(dModel)
	}
	return &DANNTransformer{
		embedding:    NewLinear(vocabSize, dModel), // One-hot to dModel
		pos_encoding: getPositionalEncoding(seqLen, dModel),
		blocks:       blocks,
		output_layer: NewLinear(dModel, vocabSize),
	}
}

func (t *DANNTransformer) Forward(input []int) ([][]float64, [][]float64) {
	// 1. One-hot encode input
	vocabSize := t.output_layer.weights.cols
	seqLen := len(input)
	one_hot_input := make([][]float64, seqLen)
	for i, id := range input {
		one_hot_input[i] = make([]float64, vocabSize)
		one_hot_input[i][id] = 1.0
	}

	// 2. Embedding + Positional Encoding
	x := t.embedding.Forward(one_hot_input)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < len(x[i]); j++ {
			x[i][j] += t.pos_encoding[i][j]
		}
	}

	// 3. Transformer Blocks
	for _, block := range t.blocks {
		x = block.Forward(x)
	}

	// 4. Final Output Layer
	return t.output_layer.Forward(x), x // Return both logits and pre-logit activations
}

// --- Training & Testing on Sequence Reversal ---

func generateData(batchSize, seqLen, vocabSize int) ([][]int, [][]int) {
	inputs := make([][]int, batchSize)
	targets := make([][]int, batchSize)
	for i := 0; i < batchSize; i++ {
		// Vocab: 0=PAD, 1=START, 2=END, 3+ are numbers
		startToken, endToken := 1, 2
		seq := make([]int, seqLen-2)
		for j := range seq {
			seq[j] = rand.Intn(vocabSize-3) + 3
		}

		inputs[i] = append([]int{startToken}, seq...)
		inputs[i] = append(inputs[i], endToken)

		reversed_seq := make([]int, len(seq))
		for j := 0; j < len(seq); j++ {
			reversed_seq[j] = seq[len(seq)-1-j]
		}
		targets[i] = append([]int{startToken}, reversed_seq...)
		targets[i] = append(targets[i], endToken)
	}
	return inputs, targets
}

func calculateLossAndGrads(logits []float64, targetToken int) (float64, []float64) {
	// Softmax
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}
	sumExp := 0.0
	probs := make([]float64, len(logits))
	for i, l := range logits {
		probs[i] = math.Exp(l - maxLogit)
		sumExp += probs[i]
	}
	for i := range probs {
		probs[i] /= sumExp
	}

	// Cross-Entropy Loss
	loss := -math.Log(probs[targetToken])

	// Gradients (derivative of softmax+loss)
	grads := make([]float64, len(logits))
	copy(grads, probs)
	grads[targetToken] -= 1

	return loss, grads
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Hyperparameters
	vocabSize := 13 // 0-2 special, 3-12 numbers
	seqLen := 6
	dModel := 32
	numBlocks := 2
	epochs := 500
	learningRate := 0.001

	fmt.Println("--- Initializing dANN-Transformer ---")
	model := NewDANNTransformer(vocabSize, seqLen, dModel, numBlocks)

	fmt.Println("--- Starting Training on Sequence Reversal ---")
	for epoch := 0; epoch < epochs; epoch++ {
		inputs, targets := generateData(1, seqLen, vocabSize)
		inputSeq := inputs[0]
		targetSeq := targets[0]

		// Forward pass
		outputLogits, final_activations := model.Forward(inputSeq)

		// --- Simplified Backpropagation & Update ---
		totalLoss := 0.0
		for i := 0; i < seqLen; i++ {
			loss, grads := calculateLossAndGrads(outputLogits[i], targetSeq[i])
			totalLoss += loss

			// Update final layer weights (simplified SGD)
			for r := 0; r < model.output_layer.weights.rows; r++ { // dModel
				for c := 0; c < model.output_layer.weights.cols; c++ { // vocabSize
					grad_for_weight := grads[c] * final_activations[i][r]
					model.output_layer.weights.data[r*vocabSize+c] -= learningRate * grad_for_weight
				}
			}
		}

		if epoch%50 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, totalLoss/float64(seqLen))
		}
	}

	fmt.Println("\n--- Training Complete ---")
	fmt.Println("--- Testing Model on a Sample ---")

	testInput, expectedOutput := generateData(1, seqLen, vocabSize)
	fmt.Printf("Input:    %v\n", testInput[0])
	fmt.Printf("Expected: %v\n", expectedOutput[0])

	finalLogits, _ := model.Forward(testInput[0])
	predictedOutput := make([]int, seqLen)
	for i, logits := range finalLogits {
		maxIdx := 0
		maxVal := logits[0]
		for j, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		predictedOutput[i] = maxIdx
	}

	fmt.Printf("Predicted: %v\n", predictedOutput)
}