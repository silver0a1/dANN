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

// --- Multi-Head Self-Attention Replacement (Simplified Dot Product Attention) ---
func SimpleSelfAttention(x []float64) []float64 {
    scale := 1.0 / math.Sqrt(float64(len(x)))
    out := make([]float64, len(x))
    for i := range x {
        out[i] = x[i] * scale
    }
    return out
}

// --- Dendritic Layer Prototype ---
// (unchanged)

// --- Transformer Block with dANN ---
// (unchanged logic, replace call to MultiHeadSelfAttention with SimpleSelfAttention)

// --- Training and Main ---
func main() {
    rand.Seed(time.Now().UnixNano())
    featureSize := 16
    numCompartments := 4
    model := NewTransformerDANNBlock(featureSize, numCompartments)

    dataset := make([]TrainingSample, 1000) // Expanded dataset
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

    validationSet := make([]TrainingSample, 200) // Validation split
    for i := range validationSet {
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
        validationSet[i] = TrainingSample{Input: input, Target: target}
    }

    lr := 0.01
    for epoch := 0; epoch < 50; epoch++ { // Increased epochs
        var totalLoss float64
        var correct int
        for _, sample := range dataset {
            rawOutput, norm2 := model.Forward(sample.Input)
            prediction := sigmoid(rawOutput)
            loss := math.Pow(prediction-sample.Target, 2)
            totalLoss += loss
            if (prediction >= 0.5 && sample.Target == 1.0) || (prediction < 0.5 && sample.Target == 0.0) {
                correct++
            }
            model.Backward(norm2, prediction, sample.Target, lr)
        }
        avgLoss := totalLoss / float64(len(dataset))
        accuracy := float64(correct) / float64(len(dataset)) * 100.0

        // Validation
        var valCorrect int
        for _, sample := range validationSet {
            rawOutput, _ := model.Forward(sample.Input)
            prediction := sigmoid(rawOutput)
            if (prediction >= 0.5 && sample.Target == 1.0) || (prediction < 0.5 && sample.Target == 0.0) {
                valCorrect++
            }
        }
        valAccuracy := float64(valCorrect) / float64(len(validationSet)) * 100.0

        fmt.Printf("Epoch %d: Avg Loss = %.4f, Train Acc = %.2f%%, Val Acc = %.2f%%\n", epoch, avgLoss, accuracy, valAccuracy)
    }
}
