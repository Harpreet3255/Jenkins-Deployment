import unittest
from model import build_model

class TestModel(unittest.TestCase):

    def test_model_architecture(self):
        model = build_model(input_dim=5000, output_dim=128, input_length=100)
        
        # Check the model has layers
        self.assertGreater(len(model.layers), 0, "Model should have layers")
        
        # Check the output shape of the model
        self.assertEqual(model.output_shape, (None, 1), "Model output shape should be (None, 1) for binary classification")

if __name__ == '__main__':
    unittest.main()
