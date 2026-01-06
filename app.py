import streamlit as st
import numpy as np
from PIL import Image

# -----------------------------
# Pooling functions
# -----------------------------
def sliding_pooling(matrix, pool_size=2, mode="max"):
    h, w, c = matrix.shape
    new_h = h - pool_size + 1
    new_w = w - pool_size + 1

    pooled = np.zeros((new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            window = matrix[i:i+pool_size, j:j+pool_size, :]

            if mode == "max":
                pooled[i, j] = np.max(window, axis=(0, 1))
            elif mode == "min":
                pooled[i, j] = np.min(window, axis=(0, 1))
            elif mode == "avg":
                pooled[i, j] = np.mean(window, axis=(0, 1))

    return pooled.astype(np.uint8)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RGB Pooling Demo", layout="wide")
st.title("🖼️ Image → RGB Matrix → 16×16 → Sliding Pooling")

uploaded_file = st.file_uploader(
    "📂 Please import an image to begin",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# ASK USER TO IMPORT IMAGE
# --------------------------------
if uploaded_file is None:
    st.info("Upload an image file to start the processing pipeline.")
    st.stop()

# --------------------------------
# PROCESS IMAGE AFTER UPLOAD
# --------------------------------
image = Image.open(uploaded_file).convert("RGB")
st.subheader("Original Image")
st.image(image, use_column_width=True)

# Convert to RGB array
rgb_array = np.array(image)

# Resize to 16x16
img_16 = image.resize((16, 16))
rgb_16 = np.array(img_16)

st.subheader("Compressed 16×16 RGB Image")
st.image(img_16, width=200)

# Pooling controls
pool_size = st.slider("Pooling window size", 2, 4, 2)
pooling_type = st.selectbox("Pooling type", ["max", "min", "avg"])

pooled_result = sliding_pooling(rgb_16, pool_size, pooling_type)
pooled_img = Image.fromarray(pooled_result)

st.subheader(f"Sliding {pooling_type.upper()} Pooling Result")
st.image(pooled_img, width=200)

# Optional matrix display
with st.expander("Show RGB matrices (for people who enjoy suffering)"):
    st.write("16×16 RGB Matrix")
    st.write(rgb_16)

    st.write("Pooled RGB Matrix")
    st.write(pooled_result)
