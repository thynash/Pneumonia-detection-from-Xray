import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from models import densenet
from utils.gradcam import GradCAM


# -------------------
# DEVICE
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# LOAD MODEL
# -------------------
@st.cache_resource
def load_model():
    model = densenet.get_model().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/densenet121.pt", map_location=device))
    model.eval()
    return model

model = load_model()
gradcam = GradCAM(model, model.features[-1])

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
st.title("🔍 Predict")

uploaded_file = st.file_uploader("Upload X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Input Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    tensor = transform(img_resized).unsqueeze(0).to(device)

    # -------------------
    # PREDICTION
    # -------------------
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    st.markdown(f"## 🧠 {prediction}")
    st.write(f"Confidence: {prob:.4f}")

    # -------------------
    # GRADCAM
    # -------------------
    heatmap = gradcam.generate(tensor)

    img_np = np.array(img_resized)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # -------------------
    # 🔴 ONLY FOR PNEUMONIA
    # -------------------
    if prediction == "PNEUMONIA":
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        cv2.circle(overlay, (x, y), 30, (0, 255, 0), 2)
        st.success("Critical region highlighted")
    else:
        st.info("No abnormal region detected")

    st.image(overlay, caption="Grad-CAM Output", use_container_width=True)

    # -------------------
    # ⚠️ IMPORTANT DISCLAIMER
    # -------------------
    st.warning("""
    Grad-CAM highlights where the model is focusing, not exact disease location.
    It provides interpretability, not precise medical segmentation.
    """)
