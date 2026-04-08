import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("AI Pipeline & Concrete Opening Detector")
st.write("Upload a pipeline or concrete image. The app will detect true cracks/openings, ignoring scratches or thin lines.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for mobile-friendly processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")

    # Enhance contrast to highlight real openings
    enhancer = ImageEnhance.Contrast(gray)
    gray_enhanced = enhancer.enhance(2.5)

    # Convert to NumPy array
    img_array = np.array(gray_enhanced, dtype=float)

    # Edge detection using Pillow
    edges = gray_enhanced.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold edges
    threshold = np.percentile(edge_array, 92)  # focus on strongest edges
    edge_binary = edge_array > threshold

    # Filter out thin lines: ignore regions smaller than min_length
    min_length = 10  # minimum consecutive pixels for true opening
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
        # Last segment
        if prev - start + 1 >= min_length:
            crack_mask[y, start:prev+1] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 500  # adjust for subtle openings

    draw = ImageDraw.Draw(image_resized)

    if crack_pixels > crack_pixel_threshold:
        # Draw solid red lines along true openings
        for y in range(height):
            x_positions = np.where(crack_mask[y, :])[0]
            if len(x_positions) > 0:
                start = x_positions[0]
                prev = x_positions[0]
                for x in x_positions[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        draw.line((start, y, prev, y), fill="red", width=2)
                        start = x
                        prev = x
                draw.line((start, y, prev, y), fill="red", width=2)

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
        file_name="opening_detection_result.png",
        mime="image/png"
    )