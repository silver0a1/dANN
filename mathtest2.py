import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define observed MSE results for each (Depth, Width)
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

# Function to compute total parameter count
def compute_param_count(input_size, hidden_size, depth, output_size=1):
    params = input_size * hidden_size + hidden_size
    for _ in range(depth - 1):
        params += hidden_size * hidden_size + hidden_size
    params += hidden_size * output_size + output_size
    return params

# Compute and add parameter count
df["Params"] = df.apply(lambda row: compute_param_count(30, int(row["Width"]), int(row["Depth"])), axis=1)

# Plot MSE vs Params (log-log)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Params", y="MSE", hue="Depth", palette="Set2", s=100)
plt.xscale("log")
plt.yscale("log")
plt.title("Performance-Efficiency Tradeoff: MSE vs. Parameter Count")
plt.xlabel("Parameter Count (Log Scale)")
plt.ylabel("Mean Squared Error (Log Scale)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("performance_efficiency_tradeoff.png")
plt.show()

# Print sorted efficiency table
print("\nSorted by MSE (best performers):")
print(df.sort_values(by="MSE")[["Depth", "Width", "Params", "MSE"]].to_string(index=False))
