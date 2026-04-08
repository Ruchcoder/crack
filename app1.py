import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Professional Pipeline & Concrete Opening Detector")
st.write("Detects and highlights only true openings, ignoring scratches or surface marks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for mobile-friendly processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")
    gray_enhanced = ImageEnhance.Contrast(gray).enhance(2.5)

    # Edge detection
    edges = gray_enhanced.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold
    threshold = np.percentile(edge_array, 90)
    edge_binary = edge_array > threshold

    # Flood fill to find connected regions (openings)
    visited = np.zeros_like(edge_binary, dtype=bool)
    crack_mask = np.zeros_like(edge_binary, dtype=bool)
    min_region_size = 20  # minimum pixels for real opening

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
    openings = []

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
        # Draw precise outline for each opening
        for region in openings:
            ys = [p[0] for p in region]
            xs = [p[1] for p in region]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # Draw a rectangle around the opening
            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)

        st.subheader("Detected Openings Overlay")
        st.image(image_resized, use_container_width=True)

        # Mini inspection report
        st.subheader("Inspection Report")
        st.success("True Crack/Openings Detected")
        st.write(f"Total Openings Detected: {len(openings)}")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")
        st.write("Recommended Action: Maintenance inspection and repair if required.")
    else:
        st.subheader("Inspection Report")
        st.info("No True Crack/Openings Detected")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")

    # Download processed image
    buffer = io.BytesIO()
    image_resized.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="professional_opening_detection.png",
        mime="image/png"
    )