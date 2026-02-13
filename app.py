import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import warnings
import tensorflow as tf

# Suppress all warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure page
st.set_page_config(page_title="Alzheimer's Detection", layout="centered")

# Title and description
st.title("🧠 Alzheimer's Detection System")
st.write("Upload an MRI brain scan image to predict Alzheimer's disease status")

st.info("""
**⚠️ Important**: This model is trained ONLY on brain MRI scans for Alzheimer's detection.
- ✅ Supported: Brain MRI scans (grayscale or color)
- ❌ Not Supported: X-rays, CT scans, other MRI types, or non-medical images
""")

# Load model
@st.cache_resource
def load_trained_model():
    if os.path.exists('cnn_model.keras'):
        return load_model('cnn_model.keras')
    else:
        st.error("Model file 'cnn_model.keras' not found. Please train the model first.")
        return None

model = load_trained_model()

# Class names and Alzheimer's status
class_names = ['healthy', 'mild', 'moderate', 'very_mild']
alzheimers_status = {
    'healthy': '✅ No Alzheimer\'s Detected',
    'mild': '⚠️ Mild Alzheimer\'s Detected',
    'moderate': '🔴 Moderate Alzheimer\'s Detected',
    'very_mild': '⚠️ Very Mild Alzheimer\'s Detected'
}

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=2)
    
    # Resize to 128x128
    img_array = np.resize(img_array, (128, 128))
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch and channel dimensions
    img_array = img_array.reshape(1, 128, 128, 1)
    
    return img_array

def validate_mri_image(image):
    """Validate if the image is likely an MRI scan"""
    img_array = np.array(image)
    
    # Check image size (should be reasonable for MRI)
    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
        return False, "Image is too small. Brain MRI scans are typically larger."
    
    # Convert to grayscale if needed
    if img_array.ndim == 3:
        # Create a grayscale version for the remaining validation steps.
        # Do NOT reject images just because they contain many colors (e.g., screenshots).
        # Convert to grayscale and continue validation based on intensity statistics.
        gray_img = np.mean(img_array, axis=2)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_diff = np.std(r) + np.std(g) + np.std(b)
        # If color pattern looks like vegetation/nature, reject; otherwise proceed.
        # (Keep the vegetation check below.)
    else:
        gray_img = img_array
    
    mean_val = np.mean(gray_img)
    std_val = np.std(gray_img)
    
    # If image is too uniform (low std), it might not be a real MRI
    if std_val < 10:
        return False, "Image lacks contrast. This might not be a valid MRI scan."
    
    # If image is extremely bright or dark
    if mean_val > 240 or mean_val < 15:
        return False, "Image appears to be invalid (too bright or too dark). Ensure you're uploading a proper MRI scan."
    
    # Check if image looks like it has natural features (green, blue colors typical of flowers/nature)
    if img_array.ndim == 3:
        # Normalize color channels
        r_norm = img_array[:,:,0] / 255.0
        g_norm = img_array[:,:,1] / 255.0
        b_norm = img_array[:,:,2] / 255.0
        
        # If green is significantly higher than red (likely vegetation/flower)
        green_avg = np.mean(g_norm)
        red_avg = np.mean(r_norm)
        if green_avg > red_avg + 0.2:
            return False, "Image appears to contain vegetation or natural objects. Please upload a brain MRI scan."
    
    return True, "Valid image"

# File uploader
uploaded_file = st.file_uploader("Choose an MRI brain scan image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

if uploaded_file is not None and model is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', width=300)
    
    # Validate if image is an MRI
    is_valid, validation_msg = validate_mri_image(image)
    
    if not is_valid:
        st.error(f"❌ Invalid Image: {validation_msg}")
        st.warning("Please upload a valid brain MRI scan image.")
    else:
        # Preprocess and make prediction
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        
        # Check if confidence is too low (model is unsure)
        if confidence < 50:
            st.warning("⚠️ Low Confidence Prediction")
            st.write(f"The model is only {confidence:.2f}% confident in its prediction.")
            st.write("This image might not be a brain MRI scan compatible with this model.")
            st.write("Please verify the image is a proper Alzheimer's detection MRI scan.")
        else:
            # Display result
            st.subheader("Prediction Result")
            
            # Status indicator
            status_color = {
                'healthy': '🟢',
                'mild': '🟡',
                'moderate': '🔴',
                'very_mild': '🟠'
            }
            
            st.markdown(f"### {status_color[predicted_class]} {alzheimers_status[predicted_class]}")
            st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show detailed probabilities
        st.subheader("Detailed Analysis")
        
        col_prob = st.columns(4)
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            with col_prob[i]:
                st.metric(class_name.capitalize(), f"{prob*100:.2f}%")
        
        # Show probability chart
        st.bar_chart({
            'Healthy': prediction[0][0] * 100,
            'Very Mild': prediction[0][3] * 100,
            'Mild': prediction[0][1] * 100,
            'Moderate': prediction[0][2] * 100
        })
        
        # Clinical interpretation
        st.subheader("Clinical Interpretation")
        
        if confidence < 50:
            st.error("❌ Unable to provide diagnosis. The uploaded image may not be a brain MRI scan related to Alzheimer's detection.")
        elif predicted_class == 'healthy':
            st.success("✅ The MRI scan shows no signs of Alzheimer's disease.")
        elif predicted_class == 'very_mild':
            st.warning("⚠️ The scan suggests very mild cognitive changes. Further monitoring recommended.")
        elif predicted_class == 'mild':
            st.warning("⚠️ The scan indicates mild Alzheimer's disease. Clinical consultation is recommended.")
        elif predicted_class == 'moderate':
            st.error("🔴 The scan shows moderate signs of Alzheimer's disease. Urgent medical attention recommended.")
        
        st.info("⚕️ **DISCLAIMER**: This model is for reference only. Please consult a medical professional for accurate diagnosis.")

else:
    if not model:
        st.error("Please train the model first by running: python cnn_classify.py")
    else:
        st.info("👆 Upload an MRI brain scan image to get started")
