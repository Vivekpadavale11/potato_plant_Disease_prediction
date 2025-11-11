import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model("potato_disease_model.h5")

# Class labels
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return {predicted_class: confidence}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🥔 Potato Disease Classification",
    description="Upload a potato leaf image to detect Early Blight, Late Blight, or Healthy leaf."
)

interface.launch()
