import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Title
st.title("Breast Cancer Detection from Mammograms")
st.write("Upload a mammogram image to check for malignancy.")

# Load model
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Function to predict
@st.cache_data
def predict(image):
    image = Image.open(image).convert("RGB")
    inputs = feature_extractor(image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    class_labels = ["Benign", "Malignant"]
    result = {class_labels[i]: float(probs[0][i]) for i in range(len(class_labels))}
    return result

# File uploader
uploaded_file = st.file_uploader("Upload a mammogram image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    with st.spinner("Analyzing Image..."):
        prediction = predict(uploaded_file)
    
    st.subheader("Prediction:")
    st.write(prediction)

