import numpy as np
import matplotlib.pyplot as plt
import os

LOG_DIR = "../activity_logs"
OUTPUT_DIR = "histograms"

def create_histograms():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find the latest epoch log files
    latest_epoch = 0
    for f in os.listdir(LOG_DIR):
        if f.endswith(".npy"):
            try:
                epoch = int(f.split('_')[0].replace('epoch', ''))
                if epoch > latest_epoch:
                    latest_epoch = epoch
            except ValueError:
                continue

    if latest_epoch == 0:
        print("No log files found.")
        return

    print(f"Processing epoch {latest_epoch}")

    # Load the sparsity data for the latest epoch
    l0_sparsity = np.load(os.path.join(LOG_DIR, f"epoch{latest_epoch}_layer0_sparsity.npy"))
    l1_sparsity = np.load(os.path.join(LOG_DIR, f"epoch{latest_epoch}_layer1_sparsity.npy"))

    # --- Dendrite-level Histograms ---
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Dendrite Activation Sparsity (Epoch {latest_epoch})')

    plt.subplot(1, 2, 1)
    plt.hist(l0_sparsity.flatten(), bins=20, color='skyblue', edgecolor='black')
    plt.title("Layer 0 Dendrites")
    plt.xlabel("% Inactive")
    plt.ylabel("Number of Dendrites")

    plt.subplot(1, 2, 2)
    plt.hist(l1_sparsity.flatten(), bins=20, color='salmon', edgecolor='black')
    plt.title("Layer 1 Dendrites")
    plt.xlabel("% Inactive")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch{latest_epoch}_dendrite_histograms.png"))
    plt.close()

    # --- Neuron-level Histograms ---
    l0_neuron_sparsity = np.mean(l0_sparsity, axis=1)
    l1_neuron_sparsity = np.mean(l1_sparsity, axis=1)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Neuron Activation Sparsity (Epoch {latest_epoch})')

    plt.subplot(1, 2, 1)
    plt.hist(l0_neuron_sparsity, bins=10, color='skyblue', edgecolor='black')
    plt.title("Layer 0 Neurons")
    plt.xlabel("Average % Inactive")
    plt.ylabel("Number of Neurons")

    plt.subplot(1, 2, 2)
    plt.hist(l1_neuron_sparsity, bins=10, color='salmon', edgecolor='black')
    plt.title("Layer 1 Neurons")
    plt.xlabel("Average % Inactive")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch{latest_epoch}_neuron_histograms.png"))
    plt.close()

    print(f"Histograms saved to '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    create_histograms()
