import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

GROQ_API_KEY = "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"  # Replace with your actual API key

# Set page config for a polished look
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load a TFLite model and return an interpreter
@st.cache_resource
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocessing function (assuming your models expect 224x224 images)
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# AI Remedy Generator using Groq
def get_ai_remedy_groq(disease_name, crop_name):
    prompt = f"Suggest step-by-step remedies for treating {disease_name} in {crop_name} plants."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an agricultural expert who gives solutions for plant diseases within 800 tokens. "},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error from Groq API: {result.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Dictionary mapping crop names to TFLite model file paths and class names
MODELS = {
    "Potato": {
        "path": os.path.join("models", "potato_model_epoch_39.tflite"),
        "classes": ["Early Blight", "Late Blight", "Healthy", "random"],
    },
    "Corn": {
        "path": os.path.join("models", "corn_model.tflite"),
        "classes": ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"],
    },
    "Apple": {
        "path": os.path.join("models", "apple_model.tflite"),
        "classes": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
    }
}

# Sidebar for crop selection
st.sidebar.title("Select Crop")
crop = st.sidebar.selectbox("Choose a crop", list(MODELS.keys()))

st.sidebar.markdown("---")
st.sidebar.info("Upload an image and get a prediction for the selected crop.")

# Load model path and class labels
model_info = MODELS[crop]
model_path = model_info["path"]
class_names = model_info["classes"]

# Title and instruction
st.title(f"{crop} Disease Detection")
st.markdown("### Limitation: Please upload only specific plant-related images.")

example_path = os.path.join("assets", "example_leaf.jpg")
st.image(example_path, caption="Example of a clear single leaf image to upload", width=300)

st.markdown("### Upload an image of the plant:")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        with st.spinner("Loading model and making prediction..."):
            interpreter = load_model(model_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            processed_image = preprocess_image(image)
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            if prediction is not None and len(prediction[0]) == len(class_names):
                confidence = np.max(prediction) * 100
                predicted_class = class_names[np.argmax(prediction)]

                st.session_state["prediction_made"] = True
                st.session_state["predicted_class"] = predicted_class
                st.session_state["confidence"] = confidence
                st.session_state["crop"] = crop

                if confidence < 85:
                    st.warning("The system cannot confidently identify this image. Please upload a clearer or different image.")
                else:
                    st.success(f"Prediction: **{predicted_class}**")
                    st.info(f"Confidence: **{confidence:.2f}%**")
            else:
                st.error("Prediction output shape mismatch with class names. Please check your model or class list.")

# Remedy suggestion after prediction
if st.session_state.get("prediction_made"):
    predicted = st.session_state.get("predicted_class")
    confidence = st.session_state.get("confidence", 0)
    crop_name = st.session_state.get("crop", "the plant")

    if confidence < 50:
        st.error("This image may not be a related plant image. Please upload a clear and relevant plant image.")

    elif confidence >= 85 and predicted.lower() != "healthy":
        if st.button("Get Remedy"):
            with st.spinner("Generating AI-powered remedy using Groq..."):
                remedy = get_ai_remedy_groq(predicted, crop_name)
                st.subheader(f"AI-Generated Remedy for {predicted}:")
                st.markdown(remedy)

# Optional styling
st.markdown("""
    <style>
        .reportview-container {
            background: #f9f9f9;
        }
        .sidebar .sidebar-content {
            background: #fff;
        }
    </style>
""", unsafe_allow_html=True)
