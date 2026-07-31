# STEP 1: Import Required Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# STEP 2: Configure the Streamlit web application's title, icon, and page layout
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺",
layout="centered")

# Class order must match training: alphabetical -> benign=0, malignant=1
CLASS_NAMES = ["benign", "malignant"]

# STEP 3: Load Saved Model and Preprocessing Objects
# Load the model from the models/ folder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/skin_lesion_classifier.keras")
    return model

# STEP 4: Write the prediction function
# Write the predict function
def predict(model, pil_image):
    """Make prediction and return probabilities"""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]   # softmax output, shape (2,)
    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx]
    prob_benign = probs[0] * 100
    prob_malignant = probs[1] * 100
    return label, prob_benign, prob_malignant

# STEP 5: Build the User Interface (UI)
st.title("🩺 Skin Lesion Classifier")
st.write("Upload a dermatoscopic image of a skin lesion to classify it as "
         "Benign or Malignant.")
st.warning("⚠️ Educational tool only (GET 324 mini-project). Not a "
           "diagnostic device — always consult a dermatologist.")

model = load_model()
uploaded_file = st.file_uploader("Upload a skin lesion image",
type=["jpg","jpeg","png"])

# STEP 6: Make predictions and display the results
# Call predict() and display the prediction result
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    label, benign_pct, malignant_pct = predict(model, img)
    st.write(f"**Prediction:** {label.capitalize()}")
    st.progress(int(malignant_pct), text=f"Malignant: {malignant_pct:.1f}%")
    st.progress(int(benign_pct), text=f"Benign: {benign_pct:.1f}%")
