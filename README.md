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

