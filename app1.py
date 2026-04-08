import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io

st.title("AI Infrastructure Crack Detection System")
st.write("Upload an image of a pipeline or concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    gray = image.convert("L")

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    threshold_pixel = 60
    edge_binary = edge_array > threshold_pixel

    crack_pixels = np.sum(edge_binary)
    crack_pixel_threshold = 2000  # min pixels to consider a real crack

    draw = ImageDraw.Draw(image)

    if crack_pixels > crack_pixel_threshold:
        height, width = edge_binary.shape
        for y in range(height):
            x_positions = np.where(edge_binary[y, :])[0]
            if len(x_positions) > 0:
                # Find continuous segments
                start = x_positions[0]
                prev = x_positions[0]
                for x in x_positions[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        # Draw line for previous segment
                        draw.line((start, y, prev, y), fill="red", width=2)
                        start = x
                        prev = x
                # Draw the last segment
                draw.line((start, y, prev, y), fill="red", width=2)

        # Determine severity
        severity = "Low"
        if crack_pixels > 5000:
            severity = "Moderate"
        if crack_pixels > 15000:
            severity = "Severe"

        st.subheader("Detected Crack Overlay")
        st.image(image, use_container_width=True)

        st.subheader("Inspection Result")
        st.success("Surface Crack Detected")
        st.write("Severity Level:", severity)
        st.write("Crack Pixel Count:", crack_pixels)

        st.subheader("Inspection Report")
        st.write("""
Defect Type: Surface Crack  
Structure: Pipeline / Concrete Surface  
Action: Maintenance Inspection Recommended
""")

    else:
        st.subheader("Inspection Result")
        st.info("No Crack Detected on Surface")
        st.write("Crack Pixel Count:", crack_pixels)

    # Download processed image
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="crack_detection_result.png",
        mime="image/png"
    )