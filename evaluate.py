import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_data

# Load model and data
model = tf.keras.models.load_model("lstm_model.h5")
df = load_data("news.csv")

# Process test data
X_test = np.array(df['text'])  # Placeholder for processed text
y_test = np.array(df['label'])

# Predict and evaluate
predictions = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
