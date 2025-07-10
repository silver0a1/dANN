# === dashboard.py (updated to visualize sparsity tracking with debug prints) ===
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔬 dANN Activity Dashboard")
st.markdown("Live visualization of dendritic and soma activation and sparsity during training.")

# --- Configuration ---
log_dir = "activity_logs"
epochs = sorted(set(
    int(fname.split("_")[0][5:])
    for fname in os.listdir(log_dir)
    if fname.endswith(".png") and "layer0" in fname
))

selected_epoch = st.slider("Select Epoch", min_value=min(epochs), max_value=max(epochs), step=5000)
col1, col2 = st.columns(2)

for layer, col in zip([0, 1], [col1, col2]):
    img_path = os.path.join(log_dir, f"epoch{selected_epoch}_layer{layer}_sparsity.png")
    if os.path.exists(img_path):
        image = Image.open(img_path)
        col.image(image, caption=f"Layer {layer} Sparsity Heatmap - Epoch {selected_epoch}")

        # === Debug print for development ===
        sparse_npy_path = os.path.join(log_dir, f"epoch{selected_epoch}_layer{layer}_sparsity.npy")
        if os.path.exists(sparse_npy_path):
            sparsity_data = np.load(sparse_npy_path)
            st.text(f"[Debug] Layer {layer} - Avg Sparsity at Epoch {selected_epoch}: {np.mean(sparsity_data):.3f}")

            # Additional delta-debug tracking (if previous exists)
            delta_file = os.path.join(log_dir, f"epoch{selected_epoch - 5000}_layer{layer}_sparsity.npy")
            if selected_epoch >= 5000 and os.path.exists(delta_file):
                prev_data = np.load(delta_file)
                delta = np.mean(np.abs(sparsity_data - prev_data))
                st.text(f"[Debug Δ] Layer {layer} - ΔSparsity vs Epoch {selected_epoch - 5000}: {delta:.6f}")
    else:
        col.warning(f"No image found for layer {layer} at epoch {selected_epoch}")
