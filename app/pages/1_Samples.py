import streamlit as st
import os
from PIL import Image

st.title("🖼️ Samples")

base_path = "dataset/test"

def show_samples(cls):
    folder = os.path.join(base_path, cls)
    files = os.listdir(folder)[:5]

    cols = st.columns(5)

    for i, file in enumerate(files):
        img = Image.open(os.path.join(folder, file))
        cols[i].image(img, use_container_width=True)

st.subheader("Normal")
show_samples("normal")

st.subheader("Pneumonia")
show_samples("pneumonia")
