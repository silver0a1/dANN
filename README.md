# Dendritic Neuron Classifier (Go)

This repository implements a biologically inspired **dendritic neuron** model in Go, capable of solving two toy classification tasks:

1. **XOR problem** (`dendritic_neuron.go`)
2. **Circle classification** (`dendritic_neuron_circle.go`)

Both examples include:
- **Multiple dendritic compartments** with tanh activations  
- **Soma integration** with sigmoid activation  
- **Adaptive learning rate** (decays 10% every 2000 epochs)  
- **Early stopping** (configurable patience & minimum error improvement)  

---

## Prerequisites

- Go 1.18 or newer  
- Git (optional, for cloning)

---

Refactor code: removed non-ASCII spaces, ensured all braces match, and normalized formatting with go fmt

Unified logging: consolidated fmt.Printf calls onto single lines using \n

Adaptive learning rate: added initLR parameter, decayed LR by 10% every 2,000 epochs

Early stopping: added patience and minDelta parameters to halt training when no improvement

Enhanced Train signature in both dendritic_neuron.go and dendritic_neuron_circle.go

XOR example (dendritic_neuron.go): updated to use adaptive LR & early stopping, formatted predictions output

Circle example (dendritic_neuron_circle.go): same enhancements plus generateCircleData() and final accuracy report