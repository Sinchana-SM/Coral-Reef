Coral Reef Classification and Feature Visualization
This project is a web-based application built with Streamlit that classifies coral reef images into three categories: healthy, bleached, or partially bleached. It also visualizes the early convolutional layer activations to give insights into what the model is 'seeing'.
🔍 Features
- Image Upload: Upload a coral image (JPG format).
- Coral Health Classification: Predicts the health status of the coral using a pre-trained VGG19 model.
- Feature Map Visualization: Displays activation maps from early CNN layers to help understand model decisions.
🧠 Model
- Architecture: VGG19
- Trained on: Custom coral reef dataset
- Output Classes:
  - Healthy
  - Bleached
  - Partially Bleached
🛠️ Requirements
Make sure you have Python 3.7+ installed.
Install dependencies
pip install -r requirements.txt
Create a requirements.txt using:
streamlit
tensorflow
Pillow
matplotlib
Model File
The application expects the model file at the following path:
./model/coral_classification_final_vgg19.h5
Please update model_path in app.py to reflect the correct path on your machine if different.
🚀 Running the App
streamlit run app.py
📁 Project Structure
.
├── app.py                          # Streamlit app script
├── model/
│   └── coral_classification_final_vgg19.h5
├── README.md
└── requirements.txt
🖼️ Example
1. Upload an image:
   - JPG format preferred
2. The app will:
   - Display the prediction label
   - Show feature maps of the first 8 layers
📌 Notes
- This project assumes the user has a trained model file (.h5).
- The uploaded image is resized to 224x224 for model input.
📄 License
MIT License. See LICENSE for details.
