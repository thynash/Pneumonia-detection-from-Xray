import streamlit as st
import os
from PIL import Image

st.title("🖼️ Samples")

base_path = "images"

def show_samples(cls):
    folder = os.path.join(base_path, cls)

    if not os.path.exists(folder):
        st.error(f"Missing folder: {folder}")
        return

    files = os.listdir(folder)[:5]

    if len(files) == 0:
        st.warning("No images found")
        return

    cols = st.columns(5)

    for i, file in enumerate(files):
        img_path = os.path.join(folder, file)

        try:
            img = Image.open(img_path)
            cols[i].image(img, use_container_width=True)
        except:
            cols[i].warning("Error loading image")

st.subheader("Normal")
show_samples("normal")

st.subheader("Pneumonia")
show_samples("pneumonia")
