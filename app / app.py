# app/app.py

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model and label encoder
model = load_model('models/poultry_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# App UI
st.title("🐔 Poultry Disease Classifier")
st.write("Provide the symptoms below to predict the disease:")

# Checkbox Inputs
fever = st.checkbox("Fever")
diarrhea = st.checkbox("Diarrhea")
lethargy = st.checkbox("Lethargy")
egg_drop = st.checkbox("Egg Production Drop")
droopy_wings = st.checkbox("Droopy Wings")

if st.button("Predict Disease"):
    # Convert checkbox inputs to binary list
    input_data = np.array([[int(fever), int(diarrhea), int(lethargy), int(egg_drop), int(droopy_wings)]])
    
    # Predict
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    disease = label_encoder.inverse_transform([predicted_class])[0]
    
    st.success(f"🧪 Predicted Disease: **{disease}**")
    
    # Optional: Add treatment suggestions
       # Optional: Add treatment suggestions
    treatments = {
        "Coccidiosis": "Use Amprolium or other anticoccidial medications.",
        "New Castle Disease": "Isolate affected birds and consult a vet for vaccination protocols.",
        "Salmonella": "Administer antibiotics and maintain hygiene.",
        "Healthy": "No disease detected. Maintain good hygiene and monitor regularly."
    }
    
    st.info(f"💊 Recommended Treatment: {treatments.get(disease, 'Consult a vet.')}")

