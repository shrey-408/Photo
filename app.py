import streamlit as st
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Efficient Pooling Function ---
def apply_pooling(matrix, pool_type, pool_size, stride):
    """
    Applies sliding window pooling.
    Note: For very large matrices in Python loops, this can be slow.
    We use a basic loop structure to demonstrate the concept clearly.
    """
    h, w = matrix.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    # Handle edge case where output dimensions are invalid
    if out_h <= 0 or out_w <= 0:
        return np.zeros((1, 1)) 

    output_matrix = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            start_row = i * stride
            start_col = j * stride
            window = matrix[start_row : start_row + pool_size, start_col : start_col + pool_size]
            
            if pool_type == 'Max':
                output_matrix[i, j] = np.max(window)
            elif pool_type == 'Min':
                output_matrix[i, j] = np.min(window)
            elif pool_type == 'Average':
                output_matrix[i, j] = np.mean(window)
                
    return output_matrix

# --- 2. Visualization Helper ---
def plot_matrix(matrix, title, show_numbers=True):
    """
    Plots the matrix. 
    - If show_numbers is True, it draws a Heatmap with numbers.
    - If show_numbers is False, it draws a standard image (better for large grids).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if show_numbers:
        # Heatmap for small grids
        sns.heatmap(matrix, annot=True, fmt=".0f" if np.max(matrix) > 1 else ".1f", 
                    cmap="viridis", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
    else:
        # Imshow for large grids (looks like an image)
        ax.imshow(matrix, cmap='gray')
        ax.axis('off')
        
    ax.set_title(title)
    return fig

# --- 3. Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🎛️ Custom Image Pooling Lab")

# --- SIDEBAR: User Controls ---
with st.sidebar:
    st.header("Settings")
    
    # 1. Grid Size Selection
    grid_option = st.selectbox(
        "Select Compressed Grid Size:",
        options=[4, 8, 16, 32, 64, 128, 256],
        index=2 # Defaults to 16
    )
    
    # 2. Pooling Method Selection
    pool_method = st.radio(
        "Select Pooling Method:",
        options=["Max", "Min", "Average"]
    )
    
    st.divider()
    
    # 3. Sliding Window Config
    pool_size = st.slider("Kernel Size (Window)", 2, 10, 2)
    stride = st.slider("Stride (Step)", 1, 10, 2)

# --- MAIN AREA ---

# Step 1: Upload
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Read Image
    original_img = Image.open(uploaded_file)
    
    # Show Conversion Info
    st.write("### 1. Processing Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(original_img, caption="Original Input", use_container_width=True)
    
    with col2:
        # Convert to RGB Array (Matrix)
        img_rgb = original_img.convert("RGB")
        rgb_array = np.array(img_rgb)
        st.write(f"**Step A: RGB Conversion**")
        st.write(f"Shape: `{rgb_array.shape}`")
        st.info("Converted to 3-Channel RGB Matrix.")

    with col3:
        # Compress to Single Matrix (Grayscale) + User Defined Grid
        img_gray = original_img.convert('L') # Convert to Single Matrix (0-255)
        img_resized = img_gray.resize((grid_option, grid_option), resample=Image.Resampling.LANCZOS)
        matrix_input = np.array(img_resized)
        
        st.write(f"**Step B: Compression**")
        st.write(f"Grid Size: `{grid_option} x {grid_option}`")
        st.info("Compressed to Single Channel Matrix.")

    st.divider()

    # Step 2: Visualization of Input vs Output
    st.write(f"### 2. Applying {pool_method} Pooling")

    # Determine if we should show numbers (Too messy for grids larger than 32x32)
    show_annotations = True if grid_option <= 32 else False

    # Apply the Math
    with st.spinner(f"Applying {pool_method} pooling..."):
        result_matrix = apply_pooling(matrix_input, pool_method, pool_size, stride)

    # Layout for Input Matrix vs Output Matrix
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader(f"Input Grid ({grid_option}x{grid_option})")
        st.pyplot(plot_matrix(matrix_input, "Compressed Input Matrix", show_numbers=show_annotations))

    with viz_col2:
        st.subheader(f"Output Grid ({result_matrix.shape[0]}x{result_matrix.shape[1]})")
        st.pyplot(plot_matrix(result_matrix, f"Result ({pool_method})", show_numbers=show_annotations))
        
        # Explanation of what happened
        st.success(f"Logic Applied: **{pool_method}**")
        if pool_method == "Max":
            st.caption("We took the brightest pixel in every window.")
        elif pool_method == "Min":
            st.caption("We took the darkest pixel in every window.")
        elif pool_method == "Average":
            st.caption("We smoothed the pixels by averaging them.")

else:
    st.info("Please upload an image to start.")
