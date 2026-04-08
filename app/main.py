import streamlit as st

st.set_page_config(
    page_title="Pneumonia AI",
    layout="wide"
)

st.title("🫁 Pneumonia Detection AI")

st.markdown("""
### 🧠 What is Pneumonia?
Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid.

---

### ⚠️ Symptoms
- Chest pain while breathing  
- Cough with phlegm  
- Fever, chills  
- Shortness of breath  

---

### 🧬 Causes
- Bacterial infections  
- Viral infections (e.g., COVID-19)  
- Fungal infections  

---

### 💊 Treatment
- Antibiotics (bacterial)  
- Rest & hydration  
- Oxygen therapy (severe cases)  

---

### 🤖 About This AI System
This application uses **Deep Learning (DenseNet121)** to:

- Detect pneumonia from chest X-rays  
- Provide confidence score  
- Highlight critical regions using Grad-CAM  

---

### 🚀 How to Use
1. Go to **Samples** → see example X-rays  
2. Go to **Predict** → upload your own X-ray  
""")
