import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\Dharukesh M\Desktop\Arunai_hack\plant_identification_model.h5')
image_path = ''
# Load and preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

# Perform prediction
def predict_animal(image_path, label_mapping):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    
    # Map model's numeric predictions to labels
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]
    
    return predicted_label, confidence

# Label mapping
label_mapping = {0: 'Healthy', 1: 'Powedery', 2: 'Rust'}

# The image_path variable is already defined, so you can use it directly.
# For example: image_path = 'test4.JPG'  (Make sure to replace it with your actual image path)

# Predict and display results
predicted_label, confidence = predict_animal(image_path, label_mapping)

# Display the input image along with prediction
image = load_img(image_path)
plt.imshow(image)
plt.axis('off')  # Remove axis
plt.title(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}")
plt.show()
