package main

import (
	"testing"
)

func TestTransformerComponents(t *testing.T) {
	dModel := 32
	numHeads := 4

	// Test LayerNorm
	ln := NewLayerNorm(dModel)
	if ln == nil {
		t.Errorf("NewLayerNorm returned nil")
	}

	// Test AddAndNorm
	an := NewAddAndNorm(dModel)
	if an == nil {
		t.Errorf("NewAddAndNorm returned nil")
	}

	// Test MultiHeadAttention
	mha := NewMultiHeadAttention(dModel, numHeads)
	if mha == nil {
		t.Errorf("NewMultiHeadAttention returned nil")
	}

	// Basic forward pass test (just to ensure it compiles and doesn't panic)
	input := make([][]float64, 5) // seqLen = 5
	for i := range input {
		input[i] = make([]float64, dModel)
	}

	_ = ln.Forward(input)
	_ = an.Forward(input, input)
	_ = mha.Forward(input, input, input)

	// Basic backward pass test (just to ensure it compiles and doesn't panic)
	_ = ln.Backward(input)
	_, _ = an.Backward(input)
	_ = mha.Backward(input)

	// Basic update test (just to ensure it compiles and doesn't panic)
	ln.Update(0.01)
	an.Update(0.01)
	mha.Update(0.01)
}
