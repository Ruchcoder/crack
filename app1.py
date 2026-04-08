import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io
from scipy.ndimage import gaussian_filter

st.title("AI Infrastructure Crack Detection System")
st.write("Upload a pipeline or concrete surface image to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    gray = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray_enhanced = enhancer.enhance(2.0)  # increase contrast

    # Convert to numpy array
    img_array = np.array(gray_enhanced, dtype=float)

    # Smooth image to remove noise
    smoothed = gaussian_filter(img_array, sigma=1)

    # Subtract smoothed image to detect edges (like local contrast)
    edges = img_array - smoothed
    edges[edges < 0] = 0  # remove negative values

    # Threshold based on percentile
    threshold = np.percentile(edges, 95)
    edge_binary = edges > threshold

    crack_pixels = np.sum(edge_binary)
    crack_pixel_threshold = 500  # smaller for subtle cracks

    draw = ImageDraw.Draw(image)

    if crack_pixels > crack_pixel_threshold:
        # Draw solid red lines for continuous segments
        height, width = edge_binary.shape
        for y in range(height):
            x_positions = np.where(edge_binary[y, :])[0]
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

        # Determine severity
        severity = "Low"
        if crack_pixels > 3000:
            severity = "Moderate"
        if crack_pixels > 10000:
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