import numpy as np
from numba import njit
import time

# --- Core dANN Logic as Numba-compatible functions ---

@njit
def forward_pass(inputs, layers_data):
    """Performs a full forward pass through the network."""
    (layer0_comp_w, layer0_comp_b, layer0_soma_w, layer0_soma_b, 
     layer1_comp_w, layer1_comp_b, layer1_soma_w, layer1_soma_b) = layers_data

    # Layer 0
    l0_comp_sums = np.dot(layer0_comp_w, inputs) + layer0_comp_b
    l0_comp_outputs = np.tanh(l0_comp_sums)
    l0_soma_sums = np.sum(l0_comp_outputs * layer0_soma_w, axis=1) + layer0_soma_b
    l0_activated_outputs = 1 / (1 + np.exp(-l0_soma_sums))

    # Layer 1 (Output)
    l1_comp_sums = np.dot(layer1_comp_w, l0_activated_outputs) + layer1_comp_b
    l1_comp_outputs = np.tanh(l1_comp_sums)
    l1_soma_sums = np.sum(l1_comp_outputs * layer1_soma_w, axis=1) + layer1_soma_b
    l1_activated_outputs = 1 / (1 + np.exp(-l1_soma_sums))

    return (l0_comp_outputs, l0_soma_sums, l0_activated_outputs, 
            l1_comp_outputs, l1_soma_sums, l1_activated_outputs)

@njit
def run_one_epoch(data, learning_rate, layers_data):
    """Runs a single epoch of training, compiled by Numba."""
    total_error = 0
    (layer0_comp_w, layer0_comp_b, layer0_soma_w, layer0_soma_b, 
     layer1_comp_w, layer1_comp_b, layer1_soma_w, layer1_soma_b) = layers_data

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    shuffled_data = data[indices]

    for i in range(shuffled_data.shape[0]):
        inputs = shuffled_data[i, :2]
        expected = shuffled_data[i, 2:]

        # Forward Pass
        (l0_comp_outputs, _, l0_activated_outputs, 
         l1_comp_outputs, _, final_network_output) = forward_pass(inputs, layers_data)
        
        total_error += np.sum((expected - final_network_output)**2)

        # Backward Pass
        # 1. Output Layer Gradients
        d_error_d_output = -(expected - final_network_output)
        d_output_d_soma_sum = final_network_output * (1 - final_network_output)
        d_loss_d_l1_soma_sum = d_error_d_output * d_output_d_soma_sum

        # 2. Hidden Layer Gradients
        d_loss_d_l0_act = np.zeros_like(l0_activated_outputs)
        for neuron_idx in range(layer0_soma_w.shape[0]):
            d_soma_sum_d_comp_out = layer1_soma_w[0]
            d_comp_out_d_comp_sum = 1 - l1_comp_outputs[0]**2
            d_comp_sum_d_input = layer1_comp_w[0, :, neuron_idx]
            effective_weight = np.sum(d_soma_sum_d_comp_out * d_comp_out_d_comp_sum * d_comp_sum_d_input)
            d_loss_d_l0_act[neuron_idx] = d_loss_d_l1_soma_sum[0] * effective_weight

        d_l0_act_d_soma_sum = l0_activated_outputs * (1 - l0_activated_outputs)
        d_loss_d_l0_soma_sum = d_loss_d_l0_act * d_l0_act_d_soma_sum

        # Weight Updates
        # 3. Update Output Layer (Layer 1)
        d_l1_soma_sum_d_soma_w = l1_comp_outputs
        layer1_soma_w -= learning_rate * d_loss_d_l1_soma_sum * d_l1_soma_sum_d_soma_w
        layer1_soma_b -= learning_rate * d_loss_d_l1_soma_sum

        d_l1_soma_sum_d_comp_out = layer1_soma_w
        d_comp_out_d_comp_sum = 1 - l1_comp_outputs**2
        d_loss_d_l1_comp_sum = (d_loss_d_l1_soma_sum * d_l1_soma_sum_d_comp_out * d_comp_out_d_comp_sum)
        
        for c_idx in range(layer1_comp_w.shape[1]):
            d_comp_sum_d_weights = l0_activated_outputs
            layer1_comp_w[0, c_idx, :] -= learning_rate * d_loss_d_l1_comp_sum[0, c_idx] * d_comp_sum_d_weights
            layer1_comp_b[0, c_idx] -= learning_rate * d_loss_d_l1_comp_sum[0, c_idx]

        # 4. Update Hidden Layer (Layer 0)
        d_l0_soma_sum_d_soma_w = l0_comp_outputs
        layer0_soma_w -= learning_rate * d_loss_d_l0_soma_sum.reshape(-1, 1) * d_l0_soma_sum_d_soma_w
        layer0_soma_b -= learning_rate * d_loss_d_l0_soma_sum

        d_l0_soma_sum_d_comp_out = layer0_soma_w
        d_comp_out_d_comp_sum = 1 - l0_comp_outputs**2
        d_loss_d_l0_comp_sum = (d_loss_d_l0_soma_sum.reshape(-1, 1) * d_l0_soma_sum_d_comp_out * d_comp_out_d_comp_sum)

        for n_idx in range(layer0_comp_w.shape[0]):
            for c_idx in range(layer0_comp_w.shape[1]):
                d_comp_sum_d_weights = inputs
                layer0_comp_w[n_idx, c_idx, :] -= learning_rate * d_loss_d_l0_comp_sum[n_idx, c_idx] * d_comp_sum_d_weights
                layer0_comp_b[n_idx, c_idx] -= learning_rate * d_loss_d_l0_comp_sum[n_idx, c_idx]

    return layers_data, total_error

class dANN:
    """A user-friendly class to manage the dANN state."""
    def __init__(self, layer_sizes, num_compartments_per_neuron):
        self.layer_sizes = layer_sizes
        self.num_compartments = num_compartments_per_neuron
        self.layers_data = self._initialize_weights()

    def _initialize_weights(self):
        l0_comp_w = np.random.uniform(-1, 1, (4, 8, 2)).astype(np.float64)
        l0_comp_b = np.random.uniform(-1, 1, (4, 8)).astype(np.float64)
        l0_soma_w = np.random.uniform(-1, 1, (4, 8)).astype(np.float64)
        l0_soma_b = np.random.uniform(-1, 1, 4).astype(np.float64)

        l1_comp_w = np.random.uniform(-1, 1, (1, 8, 4)).astype(np.float64)
        l1_comp_b = np.random.uniform(-1, 1, (1, 8)).astype(np.float64)
        l1_soma_w = np.random.uniform(-1, 1, (1, 8)).astype(np.float64)
        l1_soma_b = np.random.uniform(-1, 1, 1).astype(np.float64)
        
        return (l0_comp_w, l0_comp_b, l0_soma_w, l0_soma_b, 
                l1_comp_w, l1_comp_b, l1_soma_w, l1_soma_b)

    def train(self, data, epochs, learning_rate):
        print("Compiling training function with Numba (first-time warmup)...")
        _, _ = run_one_epoch(data[:1], learning_rate, self.layers_data)
        print("Compilation complete.")

        start_time = time.time()
        for epoch in range(epochs):
            self.layers_data, total_error = run_one_epoch(data, learning_rate, self.layers_data)
            
            if epoch > 0 and (epoch % 5000 == 0 or epoch == epochs - 1):
                avg_error = total_error / data.shape[0]
                print(f"Epoch {epoch}, Avg Error: {avg_error:.6f}")
        
        end_time = time.time()
        print(f"Numba training took: {end_time - start_time:.4f} seconds")

    def predict(self, inputs):
        _, _, _, _, _, final_output = forward_pass(inputs, self.layers_data)
        return final_output

# --- Main Execution ---
if __name__ == "__main__":
    xor_data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=np.float64)

    training_data = np.repeat(xor_data, 50, axis=0)

    print("--- Training dANN on XOR problem (Numba Accelerated v2) ---")
    ann = dANN([2, 4, 1], 8)
    ann.train(training_data, 30000, 0.05)

    print("\n--- Training Complete ---")
    print("\n--- Testing Trained dANN ---")
    correct = 0
    for row in xor_data:
        inputs = row[:2]
        expected = row[2]
        
        prediction = ann.predict(inputs)[0]
        rounded_prediction = round(prediction)
        is_correct = rounded_prediction == expected
        if is_correct:
            correct += 1
        
        print(f"Input: {inputs}, Expected: {expected}, Prediction: {prediction:.4f}, Rounded: {rounded_prediction}, Correct: {is_correct}")

    accuracy = (correct / len(xor_data)) * 100
    print(f"\nFinal Accuracy on XOR: {accuracy:.2f}%")