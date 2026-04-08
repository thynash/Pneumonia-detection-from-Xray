import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from models import densenet
from utils.gradcam import GradCAM


# -------------------
# CONFIG
# -------------------
st.set_page_config(page_title="Predict", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------
# LOAD MODEL (SAFE + CACHED)
# -------------------
@st.cache_resource
def load_model():
    model_path = "outputs/checkpoints/densenet121.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file missing")

    model = densenet.get_model().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


try:
    model = load_model()
    gradcam = GradCAM(model, model.features[-1])
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()


# -------------------
# TRANSFORM
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# -------------------
# UI
# -------------------
st.title("🔍 Pneumonia Prediction")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["png", "jpg", "jpeg"]
)


# -------------------
# PROCESS IMAGE
# -------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception:
        st.error("Invalid image file")
        st.stop()

    # 🔥 Prevent large image crashes
    if max(image.size) > 2000:
        st.warning("Large image detected. Resizing for stability.")
        image = image.resize((512, 512))

    st.image(image, caption="Input Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    tensor = transform(img_resized).unsqueeze(0).to(device)

    # -------------------
    # PREDICTION
    # -------------------
    try:
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    # -------------------
    # DISPLAY RESULT
    # -------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🧠 Prediction: {prediction}")
        st.write(f"Confidence: {prob:.4f}")

        if prediction == "PNEUMONIA":
            st.error("Abnormality detected")
        else:
            st.success("Normal case")

    # -------------------
    # GRADCAM
    # -------------------
    try:
        heatmap = gradcam.generate(tensor)

        img_np = np.array(img_resized)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8, cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

        # 🔴 ONLY FOR PNEUMONIA
        if prediction == "PNEUMONIA":
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            cv2.circle(overlay, (x, y), 30, (0, 255, 0), 2)

        with col2:
            st.image(overlay, caption="Grad-CAM Output", use_container_width=True)

    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")

    # -------------------
    # CLEAN MEMORY (IMPORTANT)
    # -------------------
    del tensor
    if 'heatmap' in locals():
        del heatmap
    if 'overlay' in locals():
        del overlay

    # -------------------
    # DISCLAIMER
    # -------------------
    st.info("""
    Grad-CAM highlights model attention, not exact disease location.
    """)

