import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Phone-Friendly Crack/Openings Detector with Severity")
st.write("Highlights real openings on pipes/concrete surfaces and shows severity level.")

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
    openings = []  # to store regions
    opening_sizes = []  # width of each opening for severity

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
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    opening_width = max(xs) - min(xs) + 1
                    opening_sizes.append(opening_width)
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

        # Determine severity based on largest opening width
        if opening_sizes:
            max_width_opening = max(opening_sizes)
            if max_width_opening < 50:
                severity = "Low"
            elif max_width_opening < 150:
                severity = "Moderate"
            else:
                severity = "High"
        else:
            severity = "Low"

        # Inspection Report
        st.subheader("Inspection Report")
        st.success("True Crack/Openings Detected")
        st.write(f"Total Openings Detected: {len(openings)}")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")
        st.write(f"Severity Level: {severity}")
        st.write("Recommended Action: Maintenance inspection and repair if required.")

    else:
        st.subheader("Inspection Result")
        st.info("No True Crack/Openings Detected")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")
        st.write("Severity Level: None")

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