import unittest
from tensorflow.keras.models import load_model
import numpy as np

class TestSavedModel(unittest.TestCase):

    def test_model_loading(self):
        # Load the model
        model = load_model('text_classification.h5')
        self.assertEqual(model.output_shape, (None, 1), "Model output shape should be (None, 1) for binary classification")

    def test_model_prediction(self):
        # Load the model
        model = load_model('text_classification.h5')
        
        # Generate dummy input data within the model's vocabulary range
        dummy_data = np.random.randint(0, 1000, size=(1, 100))
        
        # Test prediction
        prediction = model.predict(dummy_data)
        self.assertEqual(prediction.shape, (1, 1), "Prediction output should be (1, 1) for binary classification")

if __name__ == '__main__':
    unittest.main()
