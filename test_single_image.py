import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Class names
class_names = ['healthy', 'mild', 'moderate', 'very_mild']

# Load the trained model
model = load_model('cnn_model.keras')

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = plt.imread(img_path)
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    
    # Resize to 128x128
    img = np.resize(img, (128, 128))
    
    # Normalize
    img = img / 255.0
    
    # Add batch and channel dimensions
    img = img.reshape(1, 128, 128, 1)
    
    return img

def predict_image(img_path):
    """Predict the class of an image"""
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    print(f'Image: {img_path}')
    print(f'Predicted Class: {class_names[predicted_class]}')
    print(f'Confidence: {confidence:.2f}%')
    print(f'All probabilities:')
    for i, prob in enumerate(prediction[0]):
        print(f'  {class_names[i]}: {prob*100:.2f}%')

# Test on a sample image
if __name__ == '__main__':
    # Example: test on the first healthy image
    test_image = 'data/healthy/non_2.jpg'  # Replace with actual image
    
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print('Please provide a valid image path')
        print('Example usage:')
        print('  predict_image("data/healthy/non_2.jpg")')
