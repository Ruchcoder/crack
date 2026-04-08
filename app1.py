import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Pipeline Integrity Monitoring")

st.write("Upload an image of a pipe or concrete surface to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    st.subheader("Original Image")
    st.image(image)

    st.subheader("Detected Crack Patterns")
    st.image(edges)

    st.success("Possible Surface Crack Detected")