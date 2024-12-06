import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Title for the Streamlit app
st.title("Fake vs Real News Classifier")

# Load the model and tokenizer only once, this is cached to prevent reloading them on each interaction
@st.cache_resource
def load_assets():
    try:
        model = load_model('text_classification.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        st.write("Model and Tokenizer loaded successfully!")  # Check successful loading
        return model, tokenizer
    except Exception as e:
        st.error(f"An error occurred while loading the model or tokenizer: {e}")
        return None, None

# Initialize model and tokenizer
model, tokenizer = load_assets()

# User input
user_input = st.text_area("Enter news text:")

if st.button("Classify"):
    if user_input.strip():  # Ensure there is input
        try:
            # Tokenize the input
            seq = tokenizer.texts_to_sequences([user_input])
            
            if len(seq[0]) == 0:
                st.write("The input text contains no valid tokens for classification.")
            else:
                # Pad the tokenized sequence
                padded = pad_sequences(seq, maxlen=100, padding='post')
                
                # Make prediction
                prediction = model.predict(padded)[0][0]
                label = "Real" if prediction > 0.5 else "Fake"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                # Display prediction and confidence
                st.write(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.write("Please enter some text to classify.")
