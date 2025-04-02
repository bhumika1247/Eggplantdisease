import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model  

model = load_model("eggplant_disease_model.h5")  

classes = ["Healthy", "Diseased"]
recommendations = {
    "Healthy": "✅ Your eggplant is healthy! Maintain proper watering and sunlight.",
    "Diseased": "🚨 Take action! Remove infected leaves, use organic pesticides, and adjust soil nutrients."
}

st.title("🍆 Eggplant Disease Detection System")
uploaded_file = st.file_uploader("Upload an image of an eggplant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    prediction = model.predict(img_array)  
    predicted_class = "Diseased" if prediction[0][0] > 0.5 else "Healthy"  

    st.write(f"**Prediction:** {predicted_class}")  
    st.write(f"**Recommended Actions:** {recommendations[predicted_class]}")  
