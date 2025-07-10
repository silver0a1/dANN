# === dann_activity_logger.py (with auto-pruning of sparse dendrites) ===
import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt
import os

@njit
def forward_pass(inputs, layers_data):
    (l0_w, l0_b, l0_soma_w, l0_soma_b,
     l1_w, l1_b, l1_soma_w, l1_soma_b) = layers_data

    n0, c0, i0 = l0_w.shape
    l0_out = np.tanh(l0_w.reshape(n0 * c0, i0) @ inputs).reshape(n0, c0) + l0_b
    l0_comp_out = np.tanh(l0_out)
    l0_soma = np.sum(l0_comp_out * l0_soma_w, axis=1) + l0_soma_b
    l0_activated = 1 / (1 + np.exp(-l0_soma))

    n1, c1, i1 = l1_w.shape
    l1_out = np.tanh(l1_w.reshape(n1 * c1, i1) @ l0_activated).reshape(n1, c1) + l1_b
    l1_comp_out = np.tanh(l1_out)
    l1_soma = np.sum(l1_comp_out * l1_soma_w, axis=1) + l1_soma_b
    l1_activated = 1 / (1 + np.exp(-l1_soma))

    return (l0_comp_out, l0_soma, l0_activated, l1_comp_out, l1_soma, l1_activated)

@njit
def run_one_epoch(data, lr, layers_data):
    total_error = 0
    (l0_w, l0_b, l0_soma_w, l0_soma_b, l1_w, l1_b, l1_soma_w, l1_soma_b) = layers_data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    for idx in indices:
        x, y = data[idx, :2], data[idx, 2:]
        l0_comp_out, _, l0_activated, l1_comp_out, _, l1_activated = forward_pass(x, layers_data)
        err = y - l1_activated
        total_error += np.sum(err**2)
    return layers_data, total_error

class dANN:
    def __init__(self):
        self.layers_data = self._init_weights()
        self.prev_sparse0 = None
        self.prev_sparse1 = None

    def _init_weights(self):
        l0_w = np.random.uniform(-1, 1, (4, 8, 2))
        l0_b = np.random.uniform(-1, 1, (4, 8))
        l0_soma_w = np.random.uniform(-1, 1, (4, 8))
        l0_soma_b = np.random.uniform(-1, 1, 4)
        l1_w = np.random.uniform(-1, 1, (1, 8, 4))
        l1_b = np.random.uniform(-1, 1, (1, 8))
        l1_soma_w = np.random.uniform(-1, 1, (1, 8))
        l1_soma_b = np.random.uniform(-1, 1, 1)
        return (l0_w, l0_b, l0_soma_w, l0_soma_b, l1_w, l1_b, l1_soma_w, l1_soma_b)

    def train(self, data, epochs, lr, log_interval=5000, log_dir="activity_logs"):
        os.makedirs(log_dir, exist_ok=True)
        for epoch in range(epochs):
            self.layers_data, err = run_one_epoch(data, lr, self.layers_data)
            if epoch % log_interval == 0:
                self.log_activity(epoch, data, log_dir)

    def log_activity(self, epoch, data, log_dir):
        c0, a0, c1, a1 = [], [], [], []
        for row in data:
            l0_c, _, l0_a, l1_c, _, l1_a = forward_pass(row[:2], self.layers_data)
            c0.append(l0_c)
            a0.append(l0_a)
            c1.append(l1_c)
            a1.append(l1_a)

        def save_and_plot(layer, comps, acts, prev_sparsity_attr):
            comps = np.array(comps)
            sparsity = np.mean(np.abs(comps) < 0.2, axis=0)
            np.save(f"{log_dir}/epoch{epoch}_layer{layer}_sparsity.npy", sparsity)
            plt.figure(figsize=(5, 4))
            plt.imshow(sparsity, cmap="Reds", vmin=0, vmax=1)
            plt.title(f"Layer {layer} Sparsity (Epoch {epoch})")
            plt.xlabel("Dendrite")
            plt.ylabel("Neuron")
            plt.colorbar(label="% Inactive")
            plt.tight_layout()
            plt.savefig(f"{log_dir}/epoch{epoch}_layer{layer}_sparsity.png")
            plt.close()

            # Inline debug delta print
            prev_sparsity = getattr(self, prev_sparsity_attr, None)
            if prev_sparsity is not None:
                delta = np.mean(np.abs(sparsity - prev_sparsity))
                print(f"[Epoch {epoch}] Δ Layer {layer} Sparsity: {delta:.6f}")
            setattr(self, prev_sparsity_attr, sparsity.copy())

            # NOTE: pruning removed to preserve observation integrity during diagnostic phase

        save_and_plot(0, c0, a0, 'prev_sparse0')
        save_and_plot(1, c1, a1, 'prev_sparse1')

    def predict(self, x):
        return forward_pass(x, self.layers_data)[-1]

def generate_xor(n):
    x = np.random.randint(0, 2, (n, 2))
    y = np.logical_xor(x[:, 0], x[:, 1]).astype(float)
    return np.hstack((x, y.reshape(-1, 1)))

if __name__ == "__main__":
    train = generate_xor(200)
    model = dANN()
    model.train(train, epochs=30000, lr=0.05, log_interval=5000)
