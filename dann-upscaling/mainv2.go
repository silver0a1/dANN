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
    var mean float64
    for _, v := range x {
        mean += v
    }
    mean /= float64(ln.Features)
    var variance float64
    for _, v := range x {
        variance += (v - mean) * (v - mean)
    }
    variance /= float64(ln.Features)
    invStd := 1.0 / math.Sqrt(variance+ln.Epsilon)

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
    CompW           [][][]float64
    CompB           [][]float64
    SomaW           [][]float64
    SomaB           []float64
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
            dl.CompW[o][c] = make([]float64, in)
            for i := 0; i < in; i++ {
                dl.CompW[o][c][i] = rand.Float64()*2 - 1
            }
            dl.CompB[o][c] = rand.Float64()*2 - 1
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
    out := make([]float64, dl.OutFeatures)
    for o := 0; o < dl.OutFeatures; o++ {
        var somaSum float64 = dl.SomaB[o]
        for c := 0; c < dl.NumCompartments; c++ {
            var compSum float64 = dl.CompB[o][c]
            for i := 0; i < dl.InFeatures; i++ {
                compSum += dl.CompW[o][c][i] * x[i]
            }
            compOut := tanh(compSum)
            somaSum += dl.SomaW[o][c] * compOut
        }
        out[o] = somaSum
    }
    return out
}

// --- Dummy Multi-Head Self-Attention (Identity) ---
func MultiHeadSelfAttention(x []float64) []float64 {
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
    Classifier  []float64
}

func NewTransformerDANNBlock(features, numComp int) *TransformerDANNBlock {
    classifier := make([]float64, features)
    for i := range classifier {
        classifier[i] = rand.Float64()*2 - 1
    }
    return &TransformerDANNBlock{
        InFeatures: features,
        NumComp:    numComp,
        Norm1:      NewLayerNorm(features),
        DANNLayer:  NewDendriticLayer(features, features, numComp),
        Norm2:      NewLayerNorm(features),
        Classifier: classifier,
    }
}

func (tb *TransformerDANNBlock) Forward(x []float64) float64 {
    attnOut := MultiHeadSelfAttention(x)
    res1 := make([]float64, len(x))
    for i := range x {
        res1[i] = x[i] + attnOut[i]
    }
    norm1 := tb.Norm1.Forward(res1)
    dannOut := tb.DANNLayer.Forward(norm1)
    res2 := make([]float64, len(dannOut))
    for i := range dannOut {
        res2[i] = norm1[i] + dannOut[i]
    }
    norm2 := tb.Norm2.Forward(res2)
    // Final classification: dot product with linear weights
    var output float64
    for i := range norm2 {
        output += norm2[i] * tb.Classifier[i]
    }
    return output
}

type TrainingSample struct {
    Input  []float64
    Target float64
}

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
    rand.Seed(time.Now().UnixNano())
    featureSize := 16
    numCompartments := 4
    model := NewTransformerDANNBlock(featureSize, numCompartments)

    // Synthetic binary classification data
    dataset := make([]TrainingSample, 100)
    for i := range dataset {
        input := make([]float64, featureSize)
        var sum float64
        for j := range input {
            input[j] = rand.Float64()*2 - 1
            sum += input[j]
        }
        target := 0.0
        if sum > 0 {
            target = 1.0
        }
        dataset[i] = TrainingSample{Input: input, Target: target}
    }

    // Training loop
    lr := 0.01
    for epoch := 0; epoch < 10; epoch++ {
        var totalLoss float64
        for _, sample := range dataset {
            output := model.Forward(sample.Input)
            prediction := sigmoid(output)
            loss := math.Pow(prediction-sample.Target, 2)
            totalLoss += loss
        }
        avgLoss := totalLoss / float64(len(dataset))
        fmt.Printf("Epoch %d: Avg Loss = %.4f\n", epoch, avgLoss)
    }
} 
