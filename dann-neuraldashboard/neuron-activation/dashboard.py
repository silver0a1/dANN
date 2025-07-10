import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔬 dANN Activity Dashboard")
st.markdown("Live visualization of dendritic and soma activation during training.")

# --- Config ---
log_dir = "activity_logs"
epochs = sorted(set(
    int(fname.split("_")[0][5:])
    for fname in os.listdir(log_dir)
    if fname.endswith(".png")
))

# --- Epoch Slider ---
selected_epoch = st.slider("Select Epoch", min_value=min(epochs), max_value=max(epochs), step=5000)

# --- Plot Columns ---
col1, col2 = st.columns(2)

for layer in [0, 1]:
    fname = f"epoch{selected_epoch}_layer{layer}.png"
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "rb") as img_file:
        image = Image.open(img_file)
        if layer == 0:
            col1.image(image, caption=f"Layer 0 (Hidden) - Epoch {selected_epoch}")
        else:
            col2.image(image, caption=f"Layer 1 (Output) - Epoch {selected_epoch}")
