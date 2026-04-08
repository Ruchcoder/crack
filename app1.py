import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Pipeline & Concrete Opening Detector")
st.write("Upload a pipeline or concrete image. The app will highlight true cracks/openings.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for phone-friendly processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")

    # Enhance contrast to highlight real openings
    enhancer = ImageEnhance.Contrast(gray)
    gray_enhanced = enhancer.enhance(2.5)

    img_array = np.array(gray_enhanced, dtype=float)

    # Edge detection
    edges = gray_enhanced.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold edges (strongest edges only)
    threshold = np.percentile(edge_array, 92)
    edge_binary = edge_array > threshold

    # Filter out thin lines: ignore regions smaller than min_length
    min_length = 10
    height, width = edge_binary.shape
    crack_mask = np.zeros_like(edge_binary, dtype=bool)

    for y in range(height):
        x_positions = np.where(edge_binary[y, :])[0]
        if len(x_positions) == 0:
            continue
        start = x_positions[0]
        prev = x_positions[0]
        for x in x_positions[1:]:
            if x == prev + 1:
                prev = x
            else:
                if prev - start + 1 >= min_length:
                    crack_mask[y, start:prev+1] = True
                start = x
                prev = x
        if prev - start + 1 >= min_length:
            crack_mask[y, start:prev+1] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 500

    # Create overlay: fill crack area with white
    overlay = Image.new("RGB", image_resized.size)
    overlay_array = np.array(overlay)
    overlay_array[crack_mask] = [255, 255, 255]  # white crack
    overlay = Image.fromarray(overlay_array)

    # Combine original image with overlay (original dims preserved)
    image_combined = Image.blend(image_resized, overlay, alpha=0.7)

    st.subheader("Detected Openings Overlay")
    st.image(image_combined, use_container_width=True)

    if crack_pixels > crack_pixel_threshold:
        st.subheader("Inspection Result")
        st.success("True Crack/Openings Detected")
        st.write("Opening Pixel Count:", crack_pixels)
    else:
        st.subheader("Inspection Result")
        st.info("No True Crack/Openings Detected")
        st.write("Opening Pixel Count:", crack_pixels)

    # Download processed image
    buffer = io.BytesIO()
    image_combined.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="opening_detection_result.png",
        mime="image/png"
    )