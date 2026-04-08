import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Infrastructure Crack Detection & Measurement System")
st.write("Highlights real openings on pipes/concrete surfaces and measures their size in pixels.")

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

    # Flood fill to find connected regions (openings)
    visited = np.zeros_like(edge_binary, dtype=bool)
    crack_mask = np.zeros_like(edge_binary, dtype=bool)
    min_region_size = 20  # minimum pixels for a real opening
    openings = []  # store detected regions

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
            stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
        return region_pixels

    height, width = edge_binary.shape
    for y in range(height):
        for x in range(width):
            if edge_binary[y, x] and not visited[y, x]:
                pixels = flood_fill(y, x)
                if len(pixels) >= min_region_size:
                    openings.append(pixels)
                    for py, px in pixels:
                        crack_mask[py, px] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 200
    draw = ImageDraw.Draw(image_resized)

    if crack_pixels > crack_pixel_threshold:
        st.subheader("Detected Openings Overlay")

        # Draw bounding boxes and measure each opening
        opening_measurements = []
        for region in openings:
            ys = [p[0] for p in region]
            xs = [p[1] for p in region]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width_pixels = max_x - min_x + 1
            height_pixels = max_y - min_y + 1
            opening_measurements.append((width_pixels, height_pixels))
            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)

        st.image(image_resized, use_container_width=True)

        # Inspection Report
        st.subheader("Inspection Report")
        st.success("True Crack/Openings Detected")
        st.write(f"Total Openings Detected: {len(openings)}")
        for i, (w, h) in enumerate(opening_measurements, 1):
            st.write(f"Opening {i}: Width = {w}px, Height = {h}px")
        st.write("Recommended Action: Maintenance inspection and repair required.")

    else:
        st.subheader("Inspection Result")
        st.info("No True Crack/Openings Detected")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")

    # Download processed image
    buffer = io.BytesIO()
    image_resized.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="opening_detection_with_measurements.png",
        mime="image/png"
    )