
from data_loader import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# Load data
df = load_data("news.csv")

# Preprocessing steps like tokenization would go here (assuming already tokenized for simplicity)
X = np.array(df['text'])  # Placeholder for processed text
y = np.array(df['label'])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = build_model(input_dim=5000, output_dim=128, input_length=100)  # Adjust as needed
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save("lstm_model.h5")
