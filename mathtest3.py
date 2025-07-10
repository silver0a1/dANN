import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Define MLP Data
# -----------------------------
data = [
    {"Depth": 1, "Width": 4, "MSE": 0.00876},
    {"Depth": 1, "Width": 8, "MSE": 0.00523},
    {"Depth": 1, "Width": 16, "MSE": 0.00003},
    {"Depth": 1, "Width": 32, "MSE": 0.00007},
    {"Depth": 1, "Width": 64, "MSE": 0.00001},
    {"Depth": 2, "Width": 4, "MSE": 0.01694},
    {"Depth": 2, "Width": 8, "MSE": 0.00040},
    {"Depth": 2, "Width": 16, "MSE": 0.00017},
    {"Depth": 2, "Width": 32, "MSE": 0.00002},
    {"Depth": 2, "Width": 64, "MSE": 0.00000},
    {"Depth": 3, "Width": 4, "MSE": 0.02862},
    {"Depth": 3, "Width": 8, "MSE": 0.00390},
    {"Depth": 3, "Width": 16, "MSE": 0.00014},
    {"Depth": 3, "Width": 32, "MSE": 0.00001},
    {"Depth": 3, "Width": 64, "MSE": 0.00000},
]

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Compute MLP Params
# -----------------------------
def compute_mlp_params(input_size, hidden_size, depth, output_size=1):
    params = input_size * hidden_size + hidden_size
    for _ in range(depth - 1):
        params += hidden_size * hidden_size + hidden_size
    params += hidden_size * output_size + output_size
    return params

df["Params"] = df.apply(lambda row: compute_mlp_params(30, int(row["Width"]), int(row["Depth"])), axis=1)

# -----------------------------
# Step 3: Estimate dANN Params
# -----------------------------
def estimate_dann_params(
    input_size=30, hidden_neurons=15, dendrites_per_neuron=8, output_size=2
):
    # Assume each dendrite connects to ~input_size/dendrites_per_neuron features
    inputs_per_dendrite = input_size // dendrites_per_neuron  # ~3–4
    dendritic_weights = hidden_neurons * dendrites_per_neuron * inputs_per_dendrite
    soma_weights = hidden_neurons * dendrites_per_neuron
    output_weights = hidden_neurons * output_size
    output_bias = output_size
    return dendritic_weights + soma_weights + output_weights + output_bias

dann_param_count = estimate_dann_params()
dann_mse = 0.00002  # Use best observed from your dANN runs

# -----------------------------
# Step 4: Plot Tradeoff
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Params", y="MSE", hue="Depth", palette="Set2", s=100)
plt.scatter(dann_param_count, dann_mse, color='black', s=120, marker='X', label='dANN (est.)')

plt.xscale("log")
plt.yscale("log")
plt.title("Performance-Efficiency Tradeoff: MLP vs dANN")
plt.xlabel("Parameter Count (Log Scale)")
plt.ylabel("Mean Squared Error (Log Scale)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("mlp_vs_dann_tradeoff.png")
plt.show()

# -----------------------------
# Step 5: Tabular Comparison
# -----------------------------
print(f"dANN Estimated Parameter Count: {dann_param_count}")
print(f"dANN Observed MSE: {dann_mse}")
print("\nSorted MLPs by MSE:")
print(df.sort_values(by="MSE")[["Depth", "Width", "Params", "MSE"]].to_string(index=False))
