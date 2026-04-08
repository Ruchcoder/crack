import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("Crack/Openings Detector with Real-World Measurement")
st.write("Snap a picture and detect true openings, with approximate size in cm using a reference object.")

# Camera input
uploaded_file = st.camera_input("Take a picture of the pipe or concrete surface")

# Reference object width input
ref_width_cm = st.number_input("Enter the width of a reference object in the photo (cm)", min_value=0.1, value=2.0, step=0.1)

# Manual input for reference object in pixels (draw rectangle or approximate)
ref_pixels = st.number_input("Enter the reference object width in pixels (approx.)", min_value=1, value=100, step=1)

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
    threshold = np.percentile(edge_array, 90)
    edge_binary = edge_array > threshold

    # Flood fill to find openings
    visited = np.zeros_like(edge_binary, dtype=bool)
    crack_mask = np.zeros_like(edge_binary, dtype=bool)
    min_region_size = 20
    openings = []
    opening_sizes_pixels = []

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
                    opening_width_px = max(xs) - min(xs) + 1
                    opening_sizes_pixels.append(opening_width_px)
                    for py, px in pixels:
                        crack_mask[py, px] = True

    crack_pixels = np.sum(crack_mask)
    crack_pixel_threshold = 200
    draw = ImageDraw.Draw(image_resized)

    # Calculate pixels per cm
    pixels_per_cm = ref_pixels / ref_width_cm if ref_width_cm > 0 else 1

    if crack_pixels > crack_pixel_threshold:
        # Draw openings
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

        # Convert largest opening to cm
        max_opening_px = max(opening_sizes_pixels) if opening_sizes_pixels else 0
        max_opening_cm = max_opening_px / pixels_per_cm

        # Determine severity
        if max_opening_cm < 1:
            severity = "Low"
        elif max_opening_cm < 3:
            severity = "Moderate"
        else:
            severity = "High"

        st.subheader("Detected Openings Overlay")
        st.image(image_resized, use_container_width=True)

        st.subheader("Inspection Report")
        st.success("True Crack/Openings Detected")
        st.write(f"Total Openings Detected: {len(openings)}")
        st.write(f"Total Opening Pixel Count: {crack_pixels}")
        st.write(f"Largest Opening Size: {max_opening_cm:.2f} cm")
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
        file_name="opening_detection_camera_measurement.png",
        mime="image/png"
    )