import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Fake vs Real News Classifier")

try:
    # Load the model and tokenizer
    model = load_model('text_classification.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    user_input = st.text_area("Enter news text:")
    if st.button("Classify"):
        if user_input.strip():
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=100, padding='post')
            prediction = model.predict(padded)[0][0]
            label = "Real" if prediction > 0.5 else "Fake"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            st.write(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.write("Please enter some text to classify.")
except Exception as e:
    st.error(f"An error occurred: {e}")
