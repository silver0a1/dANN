import numpy as np

class DendriticCompartment:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1)

    def process(self, inputs):
        sum = np.dot(self.weights, inputs) + self.bias
        return np.tanh(sum)

class DendriticNeuron:
    def __init__(self, num_inputs, num_compartments):
        self.compartments = [DendriticCompartment(num_inputs) for _ in range(num_compartments)]
        self.soma_weights = np.random.uniform(-1, 1, num_compartments)
        self.soma_bias = np.random.uniform(-1, 1)
        self.last_basal_inputs = None
        self.last_compartment_outputs = None
        self.last_soma_output = None

    def forward(self, inputs):
        self.last_basal_inputs = inputs
        self.last_compartment_outputs = np.array([c.process(inputs) for c in self.compartments])
        self.last_soma_output = np.dot(self.soma_weights, self.last_compartment_outputs) + self.soma_bias
        return self.last_soma_output

    def train(self, d_loss_d_raw_output, learning_rate):
        soma_weight_gradients = d_loss_d_raw_output * self.last_compartment_outputs
        soma_bias_gradient = d_loss_d_raw_output

        # Propagate error to compartments using soma weights *before* the update
        error_compartment_output = d_loss_d_raw_output * self.soma_weights
        
        for i, comp in enumerate(self.compartments):
            # Tanh derivative: 1 - tanh^2(x)
            delta_compartment_sum = error_compartment_output[i] * (1 - self.last_compartment_outputs[i]**2)
            comp.weights -= learning_rate * delta_compartment_sum * self.last_basal_inputs
            comp.bias -= learning_rate * delta_compartment_sum

        # Update soma weights and bias
        self.soma_weights -= learning_rate * soma_weight_gradients
        self.soma_bias -= learning_rate * soma_bias_gradient

    def calculate_effective_weight(self, input_index):
        effective_weight = 0.0
        for i, compartment in enumerate(self.compartments):
            d_comp_output_d_comp_sum = 1 - self.last_compartment_outputs[i]**2
            comp_weight_for_input = compartment.weights[input_index]
            soma_weight_for_comp = self.soma_weights[i]
            effective_weight += soma_weight_for_comp * d_comp_output_d_comp_sum * comp_weight_for_input
        return effective_weight

class dANN:
    def __init__(self, layer_sizes, num_compartments_per_neuron):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            num_inputs = layer_sizes[i-1]
            num_neurons = layer_sizes[i]
            self.layers.append([DendriticNeuron(num_inputs, num_compartments_per_neuron) for _ in range(num_neurons)])

    def forward(self, inputs):
        raw_outputs = []
        activated_outputs = []
        current_inputs = inputs

        for i, layer in enumerate(self.layers):
            layer_raw_outputs = np.array([neuron.forward(current_inputs) for neuron in layer])
            raw_outputs.append(layer_raw_outputs)
            
            # Sigmoid activation for all layers in this classification task
            layer_activated_outputs = 1 / (1 + np.exp(-layer_raw_outputs))
            activated_outputs.append(layer_activated_outputs)
            current_inputs = layer_activated_outputs

        return raw_outputs, activated_outputs

    def train(self, data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            np.random.shuffle(data)

            for inputs, expected in data:
                raw_outputs, activated_outputs = self.forward(inputs)
                final_network_output = activated_outputs[-1]

                error = expected - final_network_output
                total_error += np.sum(error**2)

                # Backpropagation
                d_loss_d_raw_output = [None] * len(self.layers)
                
                # Output layer
                output_layer_idx = len(self.layers) - 1
                error_with_respect_to_activated_output = -(expected - final_network_output)
                derivative_of_sigmoid = final_network_output * (1 - final_network_output)
                d_loss_d_raw_output[output_layer_idx] = error_with_respect_to_activated_output * derivative_of_sigmoid

                # Hidden layers
                for l in range(len(self.layers) - 2, -1, -1):
                    d_loss_d_raw_output[l] = np.zeros(len(self.layers[l]))
                    for i in range(len(self.layers[l])):
                        error_sum = 0
                        for j, next_neuron in enumerate(self.layers[l+1]):
                            effective_weight = next_neuron.calculate_effective_weight(i)
                            error_sum += d_loss_d_raw_output[l+1][j] * effective_weight
                        
                        activated_output = activated_outputs[l][i]
                        derivative_of_sigmoid = activated_output * (1 - activated_output)
                        d_loss_d_raw_output[l][i] = error_sum * derivative_of_sigmoid

                # Update weights
                for l, layer in enumerate(self.layers):
                    for i, neuron in enumerate(layer):
                        neuron.train(d_loss_d_raw_output[l][i], learning_rate)
            
            if epoch > 0 and (epoch % 5000 == 0 or epoch == epochs - 1):
                avg_error = total_error / len(data)
                print(f"Epoch {epoch}, Avg Error: {avg_error:.6f}")


# --- Main Execution ---
if __name__ == "__main__":
    # XOR data: [inputs], expected_output
    xor_data = [
        (np.array([0, 0]), np.array([0])),
        (np.array([0, 1]), np.array([1])),
        (np.array([1, 0]), np.array([1])),
        (np.array([1, 1]), np.array([0])),
    ]

    print("--- Training dANN on XOR problem ---")
    # Network architecture: 2 inputs -> 4 hidden neurons -> 1 output
    # Each neuron has 8 dendritic compartments
    ann = dANN([2, 4, 1], 8)
    ann.train(xor_data * 50, 30000, 0.05) # Multiply data to have more training samples

    print("\n--- Training Complete ---")
    print("\n--- Testing Trained dANN ---")
    correct = 0
    for inputs, expected in xor_data:
        _, activated_outputs = ann.forward(inputs)
        prediction = activated_outputs[-1][0]
        rounded_prediction = round(prediction)
        is_correct = rounded_prediction == expected[0]
        if is_correct:
            correct += 1
        
        print(f"Input: {inputs}, Expected: {expected[0]}, Prediction: {prediction:.4f}, Rounded: {rounded_prediction}, Correct: {is_correct}")

    accuracy = (correct / len(xor_data)) * 100
    print(f"\nFinal Accuracy on XOR: {accuracy:.2f}%")
