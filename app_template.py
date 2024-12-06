import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = tf.keras.models.load_model('/ansible/roles/tdeploy_model/file/text_classification.h5')

max_words = 1000 
max_len = 10     

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

def predict_text(text):
    """Tokenize, pad, and predict the label of the input text."""
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence_padded = pad_sequences(text_sequence, maxlen=max_len, padding='post')
    prediction = model.predict(text_sequence_padded)
    label = "REAL" if prediction[0][0] >= 0.5 else "FAKE"
    confidence = float(prediction[0][0])
    return label, confidence

# Streamlit UI
st.title("Text Classification App")
st.write("Classify text as **FAKE** or **REAL** using a trained deep learning model.")

# User input
user_input = st.text_area("Enter text to classify:", "")

if st.button("Predict"):
    if user_input.strip():
        try:
            label, confidence = predict_text(user_input)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter valid text.")
