import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt
import os

# --- Core dANN Logic with Numba ---

@njit
def forward_pass(inputs, layers_data):
    (l0_w, l0_b, l0_soma_w, l0_soma_b, 
     l1_w, l1_b, l1_soma_w, l1_soma_b) = layers_data

    n0, c0, i0 = l0_w.shape
    reshaped_l0 = l0_w.reshape(n0 * c0, i0)
    l0_sums = np.dot(reshaped_l0, inputs)
    l0_comp = l0_sums.reshape(n0, c0) + l0_b
    l0_comp_out = np.tanh(l0_comp)
    l0_soma = np.sum(l0_comp_out * l0_soma_w, axis=1) + l0_soma_b
    l0_out = 1 / (1 + np.exp(-l0_soma))

    n1, c1, i1 = l1_w.shape
    reshaped_l1 = l1_w.reshape(n1 * c1, i1)
    l1_sums = np.dot(reshaped_l1, l0_out)
    l1_comp = l1_sums.reshape(n1, c1) + l1_b
    l1_comp_out = np.tanh(l1_comp)
    l1_soma = np.sum(l1_comp_out * l1_soma_w, axis=1) + l1_soma_b
    l1_out = 1 / (1 + np.exp(-l1_soma))

    return (l0_comp_out, l0_soma, l0_out,
            l1_comp_out, l1_soma, l1_out)

@njit
def run_one_epoch(data, lr, layers_data):
    total_error = 0
    (l0_w, l0_b, l0_soma_w, l0_soma_b, 
     l1_w, l1_b, l1_soma_w, l1_soma_b) = layers_data

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    shuffled_data = data[indices]

    for i in range(shuffled_data.shape[0]):
        inputs = shuffled_data[i, :2]
        expected = shuffled_data[i, 2:]

        (l0_comp_out, _, l0_out, 
         l1_comp_out, _, final_out) = forward_pass(inputs, layers_data)
        
        total_error += np.sum((expected - final_out) ** 2)

        dE_dy = -(expected - final_out)
        dy_dz = final_out * (1 - final_out)
        dL_dz = dE_dy * dy_dz

        dL_d_l0_out = np.zeros_like(l0_out)
        for j in range(l0_out.shape[0]):
            d_soma = l1_soma_w[0]
            d_comp = 1 - l1_comp_out[0]**2
            d_in = l1_w[0, :, j]
            w_eff = np.sum(d_soma * d_comp * d_in)
            dL_d_l0_out[j] = dL_dz[0] * w_eff

        d_l0_out_dz = l0_out * (1 - l0_out)
        dL_d_l0_soma = dL_d_l0_out * d_l0_out_dz

        l1_soma_w -= lr * dL_dz * l1_comp_out
        l1_soma_b -= lr * dL_dz

        dL_d_l1_comp = dL_dz * l1_soma_w * (1 - l1_comp_out ** 2)
        for c in range(l1_w.shape[1]):
            l1_w[0, c, :] -= lr * dL_d_l1_comp[0, c] * l0_out
            l1_b[0, c] -= lr * dL_d_l1_comp[0, c]

        l0_soma_w -= lr * dL_d_l0_soma.reshape(-1, 1) * l0_comp_out
        l0_soma_b -= lr * dL_d_l0_soma

        dL_d_l0_comp = dL_d_l0_soma.reshape(-1, 1) * l0_soma_w * (1 - l0_comp_out**2)
        for n in range(l0_w.shape[0]):
            for c in range(l0_w.shape[1]):
                l0_w[n, c, :] -= lr * dL_d_l0_comp[n, c] * inputs
                l0_b[n, c] -= lr * dL_d_l0_comp[n, c]

    return layers_data, total_error

class dANN:
    def __init__(self):
        self.layers_data = self._init_weights()
        self.activity_log = []

    def _init_weights(self):
        l0_w = np.random.uniform(-1, 1, (4, 8, 2)).astype(np.float64)
        l0_b = np.random.uniform(-1, 1, (4, 8)).astype(np.float64)
        l0_soma_w = np.random.uniform(-1, 1, (4, 8)).astype(np.float64)
        l0_soma_b = np.random.uniform(-1, 1, 4).astype(np.float64)
        l1_w = np.random.uniform(-1, 1, (1, 8, 4)).astype(np.float64)
        l1_b = np.random.uniform(-1, 1, (1, 8)).astype(np.float64)
        l1_soma_w = np.random.uniform(-1, 1, (1, 8)).astype(np.float64)
        l1_soma_b = np.random.uniform(-1, 1, 1).astype(np.float64)
        return (l0_w, l0_b, l0_soma_w, l0_soma_b,
                l1_w, l1_b, l1_soma_w, l1_soma_b)

    def train(self, data, epochs, lr, log_interval=5000, save_dir="activity_logs"):
        os.makedirs(save_dir, exist_ok=True)

        print("Compiling for Numba JIT warmup...")
        _ = run_one_epoch(data[:1], lr, self.layers_data)
        print("Compilation complete.")

        for epoch in range(epochs):
            self.layers_data, total_error = run_one_epoch(data, lr, self.layers_data)

            if epoch % log_interval == 0 or epoch == epochs - 1:
                self.log_activity(epoch, data, save_dir)
                avg_err = total_error / data.shape[0]
                print(f"Epoch {epoch}, Avg Error: {avg_err:.6f}")

    def log_activity(self, epoch, batch, save_dir):
        comps0, soma0, act0 = [], [], []
        comps1, soma1, act1 = [], [], []

        for row in batch:
            inputs = row[:2]
            out = forward_pass(inputs, self.layers_data)
            comps0.append(out[0])
            soma0.append(out[1])
            act0.append(out[2])
            comps1.append(out[3])
            soma1.append(out[4])
            act1.append(out[5])

        def summarize(arr_list): return np.mean(np.array(arr_list), axis=0)

        self.plot_snapshot(epoch, summarize(comps0), summarize(act0), layer=0, save_dir=save_dir)
        self.plot_snapshot(epoch, summarize(comps1), summarize(act1), layer=1, save_dir=save_dir)

    def plot_snapshot(self, epoch, dendrites, soma_out, layer, save_dir):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(dendrites, cmap='viridis', aspect='auto')
        axs[0].set_title(f"Layer {layer} Dendritic Activations (Epoch {epoch})")
        axs[0].set_ylabel("Neuron")
        axs[0].set_xlabel("Compartment")

        axs[1].bar(range(len(soma_out)), soma_out)
        axs[1].set_ylim(0, 1)
        axs[1].set_title(f"Layer {layer} Soma Outputs (Epoch {epoch})")
        axs[1].set_xlabel("Neuron")
        axs[1].set_ylabel("Activation")

        plt.tight_layout()
        path = os.path.join(save_dir, f"epoch{epoch}_layer{layer}.png")
        plt.savefig(path)
        plt.close()

    def predict(self, inputs):
        return forward_pass(inputs, self.layers_data)[-1]

def generate_xor_data(n):
    inputs = np.random.randint(0, 2, (n, 2))
    targets = np.logical_xor(inputs[:, 0], inputs[:, 1]).astype(np.float64)
    return np.hstack((inputs, targets.reshape(-1, 1)))

if __name__ == "__main__":
    train = generate_xor_data(200)
    test = generate_xor_data(100)

    model = dANN()
    model.train(train, epochs=30000, lr=0.05, log_interval=5000)

    correct = 0
    for row in test:
        pred = round(model.predict(row[:2])[0])
        correct += int(pred == row[2])
    print(f"Test Accuracy: {correct / len(test) * 100:.2f}%")
