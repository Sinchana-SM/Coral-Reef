import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model_path = r'C:\Users\sinch\OneDrive\Documents\OneDrive\Desktop\Coral_clasification_final\pro\CORAL REEFcoral_classification_final_vgg19.h5'
model = tf.keras.models.load_model(model_path)

# Mapping from numerical labels to class names
class_names = ['healthy', 'bleached', 'partially bleached']

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to make prediction and display feature maps
def predict_and_display(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(prediction[0])]
    st.write(f'Prediction: {predicted_class}')

    # Display feature maps
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(preprocessed_img)

    layer_names = [layer.name for layer in model.layers[:8]]
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        st.pyplot()

# Streamlit app
st.title('Coral Classification and Feature Visualization')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')
    predict_and_display(img)
