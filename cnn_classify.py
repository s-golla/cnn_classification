import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set paths for the dataset
healthy_dir = 'data/healthy/'
mild_dir = 'data/mild/'
moderate_dir = 'data/moderate/'
very_mild_dir = 'data/very_mild/'

# Load and preprocess the data
# This function will load images, resize them, convert to grayscale, and normalize

def load_data(directory):
    images = []
    labels = []
    for label, category in enumerate(['healthy', 'mild', 'moderate', 'very_mild']):
        category_path = os.path.join(directory, category)
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = plt.imread(img_path)
            if img.ndim == 3:  # Check if the image has color channels
                img = np.mean(img, axis=2)  # Convert to grayscale
            img = np.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the dataset
X, y = load_data('data/')

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
X_train = X_train / 255.0
X_val = X_val / 255.0

# Reshape data to add channel dimension
X_train = X_train.reshape(-1, 128, 128, 1)
X_val = X_val.reshape(-1, 128, 128, 1)

# Convert labels to one-hot encoding for multiclass
y_train = to_categorical(y_train, num_classes=4)
y_val_cat = to_categorical(y_val, num_classes=4)

# Create a CNN model
model = Sequential()
model.add(Input(shape=(128, 128, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Multiclass output (4 classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val_cat))

# Evaluate the model
predictions_probs = model.predict(X_val)
predictions = np.argmax(predictions_probs, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
f1 = f1_score(y_val, predictions, average='weighted', zero_division=0)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Print classification report
class_names = ['healthy', 'mild', 'moderate', 'very_mild']
print('\nClassification Report:')
print(classification_report(y_val, predictions, target_names=class_names))

# Print confusion matrix
print('\nConfusion Matrix:')
cm = confusion_matrix(y_val, predictions)
print(cm)

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Save the model
model.save('cnn_model.keras')
