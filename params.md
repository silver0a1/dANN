# dANN Sizing and Tuning Guide

This guide provides recommendations for sizing and tuning Dendritic Artificial Neural Networks (dANNs), highlighting key differences and considerations compared to standard Artificial Neural Networks (ANNs).

## 1. Understanding dANN Capacity

Unlike standard ANNs where a neuron performs a simple weighted sum, a dANN neuron incorporates multiple "dendritic compartments" and a "soma." Each compartment adds its own non-linearity, making a single dANN neuron significantly more powerful and capable of modeling complex relationships than a single standard ANN neuron.

This means the concept of "network size" in dANNs has an additional dimension:

*   **Depth:** Number of hidden layers.
*   **Width (Neurons):** Number of dendritic neurons per layer.
*   **Width (Compartments):** Number of compartments within each dendritic neuron.

## 2. Key Hyperparameters and Sizing Decisions

### a. Network Architecture (Layers, Neurons, Compartments)

*   **Input Layer:** Determined by the number of features in your dataset.
*   **Output Layer:** Determined by the nature of your task (e.g., 1 for regression, 1 for binary classification, N for multi-class classification).
*   **Hidden Layers (Depth):**
    *   **General:** Start with 1 or 2 hidden layers for most problems. Deeper networks can learn more abstract, hierarchical features but are harder to train.
    *   **dANN Specific:** Due to the higher capacity of individual dANN neurons, you might find that fewer hidden layers are needed compared to a standard ANN for a problem of similar complexity.
*   **Neurons per Hidden Layer (Width - Neuron Count):**
    *   **General:** Start with a moderate number (e.g., 16, 32, 64).
    *   **dANN Specific:** You might require fewer neurons per layer than a standard ANN to achieve similar performance, as each dANN neuron is inherently more powerful.
*   **Compartments per Dendritic Neuron (Width - Compartment Count):**
    *   **dANN Specific:** This is a unique and powerful hyperparameter. It controls the internal non-linearity and feature-combining capability of each neuron.
    *   **Recommendation:** Start with a moderate number (e.g., 8, 16, 32). If the network struggles to learn complex non-linear boundaries, increasing the number of compartments can be a very effective first step before adding more neurons or layers.

### b. Activation Functions

*   **Hidden Layers:**
    *   **Sigmoid:** Used in the original `dann_sine_approximation.go` and proved effective there. However, it can suffer from vanishing gradients, especially in deeper networks.
    *   **ReLU (Rectified Linear Unit):** `max(0, x)`. Generally preferred in modern deep learning for hidden layers due to its computational efficiency and ability to mitigate vanishing gradients. Often leads to faster training.
    *   **Tanh (Hyperbolic Tangent):** Used in dendritic compartments. This is a good choice for internal compartment activations as it squashes values between -1 and 1.
*   **Output Layer:**
    *   **Regression:** Linear activation (no activation function, or `f(x) = x`).
    *   **Binary Classification:** Sigmoid (output between 0 and 1, interpreted as probability).
    *   **Multi-class Classification:** Softmax (outputs a probability distribution over classes).

### c. Optimizers

*   **Stochastic Gradient Descent (SGD):** The basic optimizer used in the original `dann_sine_approximation.go`. Updates weights after each individual data point. Can be noisy but sometimes finds good minima.
*   **Mini-Batch Gradient Descent:** Updates weights after processing a small batch of data points. Offers a balance between stability and computational efficiency.
*   **Adam (Adaptive Moment Estimation):** A highly recommended adaptive optimizer. It adjusts the learning rate for each parameter individually based on past gradients. Often converges faster and is more robust to hyperparameter choices than basic SGD. It's a strong default choice for most problems.

### d. Learning Rate

*   **General:** This is one of the most critical hyperparameters. Too high, and the model might overshoot the minimum; too low, and training will be very slow.
*   **Recommendation:** Start with values like `0.01`, `0.001`, or `0.0001`.
*   **Adam Specific:** Adam is less sensitive to the initial learning rate, but `0.001` is a common and effective starting point.
*   **Learning Rate Schedules:** Consider reducing the learning rate over time (e.g., step decay, exponential decay) to allow for finer adjustments as training progresses.

### e. Epochs and Batch Size

*   **Epochs:** One epoch means the model has seen the entire training dataset once.
    *   **General:** The optimal number varies wildly. Don't be afraid of high epoch counts (e.g., 10,000s or 100,000s) for complex problems or when using small learning rates. The original `dann_sine_approximation.go` uses 20,000 epochs effectively.
    *   **Early Stopping:** Implement early stopping by monitoring performance on a validation set. Stop training when validation error stops improving or starts increasing to prevent overfitting and save computation.
*   **Batch Size:** The number of samples processed before the model's weights are updated.
    *   **SGD:** Batch size of 1.
    *   **Mini-Batching:** Common values are powers of 2 (e.g., 32, 64, 128, 256). Larger batch sizes can lead to faster training per epoch but might converge to less optimal solutions. Smaller batch sizes provide more frequent updates and can escape local minima better but are computationally less efficient per update.

## 3. General Tuning Strategy

1.  **Start Simple:** Begin with a relatively small dANN (e.g., 1-2 hidden layers, 16-32 neurons per layer, 8-16 compartments per neuron).
2.  **Choose a Robust Optimizer:** Adam is a good default.
3.  **Use Appropriate Activations:** ReLU for hidden layers, linear for regression output, sigmoid/softmax for classification.
4.  **Establish a Baseline:** Train the initial model and record its performance (RMSE, R-squared, accuracy).
5.  **Iterative Tuning:**
    *   **Learning Rate:** Experiment with different learning rates (e.g., `0.01`, `0.001`, `0.0001`).
    *   **Capacity (Compartments/Neurons):** If the model is underfitting (high training and test error), increase the number of compartments per neuron, then neurons per layer, then add more layers.
    *   **Regularization (Epochs/Early Stopping):** If the model is overfitting (low training error, high test error), reduce epochs, implement early stopping, or consider adding regularization techniques (though not covered in this guide).
    *   **Batch Size:** Experiment with different batch sizes to find a balance between training speed and convergence quality.
6.  **Monitor Progress:** Keep track of training and validation metrics (loss, accuracy) over epochs to identify overfitting or underfitting.

By systematically exploring these hyperparameters, you can effectively size and tune your dANNs for various tasks.
