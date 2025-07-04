package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- Utility Functions & Structs ---

func newMatrix(rows, cols int, random bool) []float64 {
	data := make([]float64, rows*cols)
	if random {
		for i := range data {
			// Use Xavier initialization for better gradients
			data[i] = rand.NormFloat64() * math.Sqrt(1.0/float64(cols))
		}
	}
	return data
}

// --- Module Interface ---
// Defines the contract for any trainable part of our network.

type Module interface {
	Forward(input [][]float64) [][]float64
	Backward(dOutput [][]float64) [][]float64
	Update(learningRate float64)
}

// --- Linear Layer ---

type Linear struct {
	weights         []float64
	feedbackWeights []float64 // Fixed random weights for FA
	biases          []float64
	grads           []float64
	lastInput       [][]float64
	lastOutput      [][]float64
	inFeatures, outFeatures int
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	return &Linear{
		weights:         newMatrix(inFeatures, outFeatures, true),
		feedbackWeights: newMatrix(inFeatures, outFeatures, true),
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
	// Calculate gradients for weights using the forward pass activations and output error
	for i := 0; i < l.inFeatures; i++ {
		for j := 0; j < l.outFeatures; j++ {
			grad := 0.0
			for k := 0; k < len(dOutput); k++ {
				grad += l.lastInput[k][i] * dOutput[k][j]
			}
			l.grads[i*l.outFeatures+j] = grad
		}
	}

	// Propagate error backward using FIXED RANDOM feedback weights
	dInput := make([][]float64, len(dOutput))
	for i := 0; i < len(dOutput); i++ {
		dInput[i] = make([]float64, l.inFeatures)
		for j := 0; j < l.inFeatures; j++ {
			sum := 0.0
			for k := 0; k < l.outFeatures; k++ {
				// Note: Using feedbackWeights, not weights
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
}

// --- dANN FFN (as a Module) ---
// A simplified dANN for clarity in the backward pass

type DANN_FFN struct {
	hidden *Linear
	output *Linear
}

func NewDANN_FFN(dModel int) *DANN_FFN {
	// Simplified two-layer MLP structure for the dANN FFN
	// A more complex dendritic structure would require a more complex backward pass.
	return &DANN_FFN{
		hidden: NewLinear(dModel, dModel*4),
		output: NewLinear(dModel*4, dModel),
	}
}

func (d *DANN_FFN) Forward(input [][]float64) [][]float64 {
	hidden_out := d.hidden.Forward(input)
	// Apply ReLU activation
	for i := range hidden_out {
		for j := range hidden_out[i] {
			if hidden_out[i][j] < 0 {
				hidden_out[i][j] = 0
			}
		}
	}
	d.hidden.lastOutput = hidden_out // Store the output for the backward pass
	return d.output.Forward(hidden_out)
}

func (d *DANN_FFN) Backward(dOutput [][]float64) [][]float64 {
	dHidden := d.output.Backward(dOutput)
	// Backward pass for ReLU
	for i := range dHidden {
		for j := range dHidden[i] {
			if d.hidden.lastOutput[i][j] <= 0 { // Check the output of the hidden layer, not the input
				dHidden[i][j] = 0
			}
		}
	}
	return d.hidden.Backward(dHidden)
}

func (d *DANN_FFN) Update(learningRate float64) {
	d.hidden.Update(learningRate)
	d.output.Update(learningRate)
}


// --- Other Transformer Components (simplified) ---

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

// ... LayerNorm, SelfAttention etc. would be implemented as Modules as well ...
// For brevity, we will simplify the Transformer block logic in this example.

// --- Transformer Block ---

type TransformerBlock struct {
	attention Module // Placeholder for a full attention module
	ffn       Module
	// LayerNorm and Residual connections would be here
}

func NewTransformerBlock(dModel int) *TransformerBlock {
	return &TransformerBlock{
		attention: NewLinear(dModel, dModel), // Simplified attention
		ffn:       NewDANN_FFN(dModel),
	}
}

func (b *TransformerBlock) Forward(input [][]float64) [][]float64 {
	// Simplified: pass through attention then FFN
	attention_out := b.attention.Forward(input)
	return b.ffn.Forward(attention_out)
}

func (b *TransformerBlock) Backward(dOutput [][]float64) [][]float64 {
	d_ffn_out := b.ffn.Backward(dOutput)
	return b.attention.Backward(d_ffn_out)
}

func (b *TransformerBlock) Update(learningRate float64) {
	b.attention.Update(learningRate)
	b.ffn.Update(learningRate)
}

// --- Full Transformer Model ---

type DANNTransformer struct {
	embedding    *Linear
	pos_encoding [][]float64
	blocks       []Module
	output_layer *Linear
}

func NewDANNTransformer(vocabSize, seqLen, dModel, numBlocks int) *DANNTransformer {
	blocks := make([]Module, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = NewTransformerBlock(dModel)
	}
	return &DANNTransformer{
		embedding:    NewLinear(vocabSize, dModel),
		pos_encoding: getPositionalEncoding(seqLen, dModel),
		blocks:       blocks,
		output_layer: NewLinear(dModel, vocabSize),
	}
}

func (t *DANNTransformer) Forward(input []int) [][]float64 {
	vocabSize := t.embedding.inFeatures
	one_hot_input := make([][]float64, len(input))
	for i, id := range input {
		one_hot_input[i] = make([]float64, vocabSize)
		one_hot_input[i][id] = 1.0
	}

	x := t.embedding.Forward(one_hot_input)
	for i := range x {
		for j := range x[i] {
			x[i][j] += t.pos_encoding[i][j]
		}
	}

	for _, block := range t.blocks {
		x = block.Forward(x)
	}
	return t.output_layer.Forward(x)
}

func (t *DANNTransformer) Backward(dOutput [][]float64) {
	dx := t.output_layer.Backward(dOutput)
	for i := len(t.blocks) - 1; i >= 0; i-- {
		dx = t.blocks[i].Backward(dx)
	}
	t.embedding.Backward(dx)
}

func (t *DANNTransformer) Update(learningRate float64) {
	t.embedding.Update(learningRate)
	for _, block := range t.blocks {
		block.Update(learningRate)
	}
	t.output_layer.Update(learningRate)
}

// --- Training & Testing ---

func generateData(seqLen, vocabSize int) ([]int, []int) {
	startToken, endToken := 1, 2
	seq := make([]int, seqLen-2)
	for j := range seq {
		seq[j] = rand.Intn(vocabSize-3) + 3
	}
	input := append([]int{startToken}, seq...)
	input = append(input, endToken)

	reversed_seq := make([]int, len(seq))
	for j := 0; j < len(seq); j++ {
		reversed_seq[j] = seq[len(seq)-1-j]
	}
	target := append([]int{startToken}, reversed_seq...)
	target = append(target, endToken)
	return input, target
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Hyperparameters
	vocabSize := 10
	seqLen := 5
	dModel := 16
	numBlocks := 2
	epochs := 50000
	learningRate := 0.0001

	fmt.Println("--- Initializing dANN-Transformer with Feedback Alignment ---")
	model := NewDANNTransformer(vocabSize, seqLen, dModel, numBlocks)

	fmt.Println("--- Starting Training on Sequence Reversal ---")
	for epoch := 0; epoch < epochs; epoch++ {
		inputSeq, targetSeq := generateData(seqLen, vocabSize)

		// Forward pass
		logits := model.Forward(inputSeq)

		// Calculate loss and initial gradients
		totalLoss := 0.0
		dLoss_dLogits := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			// Softmax
			probs := make([]float64, vocabSize)
			maxLogit := logits[i][0]
			for _, l := range logits[i] {
				if l > maxLogit {
					maxLogit = l
				}
			}
			sumExp := 0.0
			for j, l := range logits[i] {
				probs[j] = math.Exp(l - maxLogit)
				sumExp += probs[j]
			}
			for j := range probs {
				probs[j] /= sumExp
			}

			totalLoss += -math.Log(probs[targetSeq[i]])

			// Gradient of loss w.r.t. logits
			dLoss_dLogits[i] = make([]float64, vocabSize)
			copy(dLoss_dLogits[i], probs)
			dLoss_dLogits[i][targetSeq[i]] -= 1
		}

		// Backward pass
		model.Backward(dLoss_dLogits)

		// Update weights
		model.Update(learningRate)

		if epoch%500 == 0 {
			fmt.Printf("Epoch %d, Avg Loss: %f\n", epoch, totalLoss/float64(seqLen))
		}
	}

	fmt.Println("\n--- Training Complete ---")
	fmt.Println("--- Testing Model on a Sample ---")

	input, expected := generateData(seqLen, vocabSize)
	fmt.Printf("Input:    %v\n", input)
	fmt.Printf("Expected: %v\n", expected)

	finalLogits := model.Forward(input)
	predicted := make([]int, seqLen)
	for i, l := range finalLogits {
		maxIdx := 0
		maxVal := -1e9
		for j, v := range l {
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		predicted[i] = maxIdx
	}

	fmt.Printf("Predicted: %v\n", predicted)
}
