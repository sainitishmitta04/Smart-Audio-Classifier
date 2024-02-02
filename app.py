import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import IPython.display as ipd
import os
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
model = load_model('model.h5')

# Load the fitted LabelEncoder
labelencoder = joblib.load('labelencoder.joblib')

# Function to extract features and make predictions
def predict(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]

# Streamlit app
st.title("Audio Classification App")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

# Display the uploaded audio file
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Predict the class on button click
    if st.button("Predict"):
        # Perform prediction
        prediction = predict(uploaded_file)

        # Display the prediction result
        st.success(f"The predicted class is: {prediction}")
