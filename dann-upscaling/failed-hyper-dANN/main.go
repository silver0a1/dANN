package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
)

// --- Utility: Layer Normalization ---
type LayerNorm struct {
    Features int
    Epsilon  float64
    Gamma    []float64
    Beta     []float64
}

func NewLayerNorm(features int) *LayerNorm {
    gamma := make([]float64, features)
    beta := make([]float64, features)
    for i := range gamma {
        gamma[i] = 1.0
        beta[i] = 0.0
    }
    return &LayerNorm{Features: features, Epsilon: 1e-5, Gamma: gamma, Beta: beta}
}

func (ln *LayerNorm) Forward(x []float64) []float64 {
    // Compute mean
    var mean float64
    for _, v := range x {
        mean += v
    }
    mean /= float64(ln.Features)
    // Compute variance
    var variance float64
    for _, v := range x {
        variance += (v - mean) * (v - mean)
    }
    variance /= float64(ln.Features)
    invStd := 1.0 / math.Sqrt(variance+ln.Epsilon)

    // Normalize
    out := make([]float64, ln.Features)
    for i, v := range x {
        norm := (v - mean) * invStd
        out[i] = norm*ln.Gamma[i] + ln.Beta[i]
    }
    return out
}

// --- Dendritic Layer Prototype ---
type DendriticLayer struct {
    InFeatures      int
    OutFeatures     int
    NumCompartments int
    CompW           [][][]float64 // [out][comp][in]
    CompB           [][]float64   // [out][comp]
    SomaW           [][]float64   // [out][comp]
    SomaB           []float64     // [out]
}

func NewDendriticLayer(in, out, numComp int) *DendriticLayer {
    dl := &DendriticLayer{
        InFeatures:      in,
        OutFeatures:     out,
        NumCompartments: numComp,
        CompW:           make([][][]float64, out),
        CompB:           make([][]float64, out),
        SomaW:           make([][]float64, out),
        SomaB:           make([]float64, out),
    }
    for o := 0; o < out; o++ {
        dl.CompW[o] = make([][]float64, numComp)
        dl.CompB[o] = make([]float64, numComp)
        dl.SomaW[o] = make([]float64, numComp)
        for c := 0; c < numComp; c++ {
            // Initialize compartment weights and bias
            dl.CompW[o][c] = make([]float64, in)
            for i := 0; i < in; i++ {
                dl.CompW[o][c][i] = rand.Float64()*2 - 1
            }
            dl.CompB[o][c] = rand.Float64()*2 - 1
            // Initialize soma weights
            dl.SomaW[o][c] = rand.Float64()*2 - 1
        }
        dl.SomaB[o] = rand.Float64()*2 - 1
    }
    return dl
}

func tanh(x float64) float64 {
    return math.Tanh(x)
}

func (dl *DendriticLayer) Forward(x []float64) []float64 {
    // Single token forward pass
    out := make([]float64, dl.OutFeatures)
    for o := 0; o < dl.OutFeatures; o++ {
        var somaSum float64 = dl.SomaB[o]
        for c := 0; c < dl.NumCompartments; c++ {
            // compute compartment output
            var compSum float64 = dl.CompB[o][c]
            for i := 0; i < dl.InFeatures; i++ {
                compSum += dl.CompW[o][c][i] * x[i]
            }
            compOut := tanh(compSum)
            somaSum += dl.SomaW[o][c] * compOut
        }
        out[o] = somaSum // raw soma sum; no activation here
    }
    return out
}

// --- Dummy Multi-Head Self-Attention (Identity) ---
func MultiHeadSelfAttention(x []float64) []float64 {
    // Placeholder: identity mapping
    out := make([]float64, len(x))
    copy(out, x)
    return out
}

// --- Transformer Block with dANN ---
type TransformerDANNBlock struct {
    InFeatures  int
    NumComp     int
    Norm1       *LayerNorm
    DANNLayer   *DendriticLayer
    Norm2       *LayerNorm
}

func NewTransformerDANNBlock(features, numComp int) *TransformerDANNBlock {
    return &TransformerDANNBlock{
        InFeatures: features,
        NumComp:    numComp,
        Norm1:      NewLayerNorm(features),
        DANNLayer:  NewDendriticLayer(features, features, numComp),
        Norm2:      NewLayerNorm(features),
    }
}

func (tb *TransformerDANNBlock) Forward(x []float64) []float64 {
    // Self-Attention + residual
    attnOut := MultiHeadSelfAttention(x)
    res1 := make([]float64, len(x))
    for i := range x {
        res1[i] = x[i] + attnOut[i]
    }
    norm1 := tb.Norm1.Forward(res1)

    // dANN block + residual
    dannOut := tb.DANNLayer.Forward(norm1)
    res2 := make([]float64, len(dannOut))
    for i := range dannOut {
        res2[i] = norm1[i] + dannOut[i]
    }
    norm2 := tb.Norm2.Forward(res2)
    return norm2
}

func main() {
    rand.Seed(time.Now().UnixNano())

    // Example: single token vector
    featureSize := 16
    numCompartments := 4
    input := make([]float64, featureSize)
    for i := range input {
        input[i] = rand.Float64()*2 - 1
    }

    block := NewTransformerDANNBlock(featureSize, numCompartments)
    output := block.Forward(input)

    fmt.Println("Input:", input)
    fmt.Println("Output:", output)
}
