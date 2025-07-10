import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---- Step 1: Generate synthetic data ----
N_SAMPLES = 100
INPUT_SIZE = 30

X = np.random.rand(N_SAMPLES, INPUT_SIZE)
X_tensor = torch.tensor(X, dtype=torch.float32)

# Simulate a single dANN neuron with 8 dendritic compartments
def simulate_dann_batch(X_batch):
    outputs = []
    for x in X_batch:
        compartments = np.array_split(x, 8)
        weights = [np.random.randn(len(c)) for c in compartments]
        dendritic_outs = [1 / (1 + np.exp(-np.dot(w, c))) for w, c in zip(weights, compartments)]
        soma_weights = np.random.randn(len(dendritic_outs))
        soma_input = np.dot(soma_weights, dendritic_outs)
        outputs.append(1 / (1 + np.exp(-soma_input)))
    return np.array(outputs)

y = simulate_dann_batch(X)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ---- Step 2: Define MLP architecture ----
class MLP(nn.Module):
    def __init__(self, input_size=30, hidden_sizes=[32], output_size=1):
        super(MLP, self).__init__()
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# ---- Step 3: Train MLPs of varying sizes ----
depths = [1, 2, 3]
widths = [4, 8, 16, 32, 64]
results = {}

for d in depths:
    for w in widths:
        hidden_sizes = [w] * d
        model = MLP(hidden_sizes=hidden_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(300):
            pred = model(X_tensor)
            loss = F.mse_loss(pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_pred = model(X_tensor).detach().numpy()
        mse = mean_squared_error(y, final_pred)
        results[(d, w)] = mse
        print(f"Depth: {d}, Width: {w}, MSE: {mse:.5f}")

# ---- Step 4: Visualize Results ----
data = []
for (d, w), mse in results.items():
    data.append({"Depth": d, "Width": w, "MSE": mse})
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Width", y="MSE", hue="Depth", marker="o")
plt.title("MLP Approximation Error vs. Network Size")
plt.ylabel("Mean Squared Error (vs. dANN Output)")
plt.xlabel("Width per Layer")
plt.grid(True)
plt.tight_layout()
plt.show()
