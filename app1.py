import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Phone-Friendly Crack/Openings Detector")
st.write("Highlights only real openings on pipes or concrete surfaces without external dependencies.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for phone-friendly processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")
    gray_enhanced = ImageEnhance.Contrast(gray).enhance(2.5)

    # Edge detection
    edges = gray_enhanced.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold strong edges
    threshold = np.percentile(edge_array, 90)
    edge_binary = edge_array > threshold

    # Simple connected region detection using flood fill
    visited = np.zeros_like(edge_binary, dtype=bool)
    crack_mask = np.zeros_like(edge_binary, dtype=bool)
    min_region_size = 20  # minimum pixels for a real opening

    def flood_fill(y, x):
        stack = [(y, x)]
        region_pixels = []
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= edge_binary.shape[0] or cx < 0 or cx >= edge_binary.shape[1]:
                continue
            if visited[cy, cx] or not edge_binary[cy, cx]:
                continue
            visited[cy, cx] = True
            region_pixels.append((cy, cx))
            # Check 4 neighbors
            stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
        return region_pixels

    height, width = edge_binary.shape
    for y in range(height):
        for x in range(width):
            if edge_binary[y, x] and not visited[y, x]:
                pixels = flood_fill(y, x)
                if len(pixels) >= min_region_size:
                    for py, px in pixels:
                        crack_mask[py, px] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 200

    draw = ImageDraw.Draw(image_resized)

    if crack_pixels > crack_pixel_threshold:
        # Draw solid red lines along detected openings
        for y in range(height):
            x_positions = np.where(crack_mask[y, :])[0]
            if len(x_positions) > 0:
                start = x_positions[0]
                prev = x_positions[0]
                for x in x_positions[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        draw.line((start, y, prev, y), fill="red", width=3)
                        start = x
                        prev = x
                draw.line((start, y, prev, y), fill="red", width=3)

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
        file_name="phone_friendly_opening_detection.png",
        mime="image/png"
    )