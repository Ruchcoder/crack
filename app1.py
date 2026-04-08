import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io
from scipy.ndimage import label, find_objects

st.title("Focused Crack/Openings Detector")
st.write("This app highlights only true openings or gaps on the surface, ignoring scratches or minor marks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for mobile-friendly processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")

    # Enhance contrast
    gray_enhanced = ImageEnhance.Contrast(gray).enhance(2.5)

    # Edge detection
    edges = gray_enhanced.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold
    threshold = np.percentile(edge_array, 92)
    edge_binary = edge_array > threshold

    # Connected components
    labeled, num_features = label(edge_binary)
    slices = find_objects(labeled)

    crack_mask = np.zeros_like(edge_binary, dtype=bool)
    min_size = 20  # minimum pixel size for a real opening

    for i, sl in enumerate(slices):
        region = labeled[sl] == (i + 1)
        if np.sum(region) >= min_size:
            crack_mask[sl][region] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 500

    draw = ImageDraw.Draw(image_resized)

    if crack_pixels > crack_pixel_threshold:
        # Draw boundary of each opening region
        height, width = crack_mask.shape
        for y in range(height):
            x_positions = np.where(crack_mask[y, :])[0]
            if len(x_positions) > 0:
                start = x_positions[0]
                prev = x_positions[0]
                for x in x_positions[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        draw.line((start, y, prev, y), fill="white", width=2)
                        start = x
                        prev = x
                draw.line((start, y, prev, y), fill="white", width=2)

        st.subheader("Detected Openings Overlay")
        st.image(image_resized, use_container_width=True)
        st.subheader("Inspection Result")
        st.success("True Crack/Openings Detected")
        st.write("Opening Pixel Count:", crack_pixels)
    else:
        st.subheader("Inspection Result")
        st.info("No True Crack/Openings Detected")
        st.write("Opening Pixel Count:", crack_pixels)

    # Download processed image
    buffer = io.BytesIO()
    image_resized.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="focused_opening_detection.png",
        mime="image/png"
    )