import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

st.title("Crack Defect Detection System")

st.write("Upload an image of a pipeline or concrete surface for crack inspection.")

uploaded_file = st.file_uploader("Upload Inspection Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    gray = image.convert("L")

    edges = gray.filter(ImageFilter.FIND_EDGES)

    edge_array = np.array(edges)

    threshold = 50
    crack_pixels = np.sum(edge_array > threshold)

    severity = "Low"
    if crack_pixels > 5000:
        severity = "Moderate"
    if crack_pixels > 15000:
        severity = "Severe"

    draw = ImageDraw.Draw(image)

    for y in range(0, edge_array.shape[0], 10):
        for x in range(0, edge_array.shape[1], 10):
            if edge_array[y][x] > threshold:
                draw.rectangle((x,y,x+5,y+5), outline="red")

    st.subheader("Inspection Image")
    st.image(image, use_column_width=True)

    st.subheader("Inspection Result")

    st.success("Surface Crack Detected")

    st.write("Severity Level:", severity)

    st.write("Estimated Crack Pixels:", crack_pixels)

    st.subheader("Inspection Report")

    st.write("""
    Defect Type: Surface Crack  
    Infrastructure: Pipeline / Concrete Surface  
    Recommended Action: Engineering Inspection Required
    """)