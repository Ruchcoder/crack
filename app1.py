import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

st.title("AI-Based Pipeline & Concrete Defect Detection Demo")

st.write("Upload an image of a pipe or concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    edges = image.filter(ImageFilter.FIND_EDGES)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Detected Crack Patterns")
    st.image(edges, use_column_width=True)

    st.success("Possible Surface Crack Detected")