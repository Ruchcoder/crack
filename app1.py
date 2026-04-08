import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
from scipy.ndimage import binary_dilation, label

st.title("AI Infrastructure Crack Detection System")
st.write("Upload an image of a pipeline or concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    gray = image.convert("L")

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    # Threshold strong edges
    threshold_pixel = 60
    edge_binary = edge_array > threshold_pixel

    # Optional: make crack lines thicker
    edge_binary = binary_dilation(edge_binary, structure=np.ones((2,2)))

    # Label connected crack segments
    labeled_array, num_features = label(edge_binary)
    crack_pixels = np.sum(edge_binary)

    crack_pixel_threshold = 2000  # min pixels to consider a real crack

    draw = ImageDraw.Draw(image)

    if crack_pixels > crack_pixel_threshold and num_features > 0:
        # Draw solid red lines along cracks
        ys, xs = np.where(edge_binary)
        for y, x in zip(ys, xs):
            draw.line((x, y, x+1, y+1), fill="red", width=2)

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
        st.write("Number of Crack Segments:", num_features)

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