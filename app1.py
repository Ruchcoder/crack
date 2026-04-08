import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io

st.title("Crack Defect Detection System")
st.write("Upload an image of a pipeline or concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    gray = image.convert("L")

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Count strong edges
    threshold_pixel = 60
    crack_pixels = np.sum(edge_array > threshold_pixel)

    # Set a minimum number of pixels to consider as a crack
    crack_pixel_threshold = 2000  # you can adjust based on image size

    draw = ImageDraw.Draw(image)

    if crack_pixels > crack_pixel_threshold:
        severity = "Low"
        if crack_pixels > 5000:
            severity = "Moderate"
        if crack_pixels > 15000:
            severity = "Severe"

        # Draw red lines on crack pixels
        height, width = edge_array.shape
        for y in range(height):
            for x in range(width):
                if edge_array[y][x] > threshold_pixel:
                    draw.line((x, y, x+1, y+1), fill="red", width=1)

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

    # Provide downloadable image regardless
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="crack_detection_result.png",
        mime="image/png"
    )