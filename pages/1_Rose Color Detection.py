import time
import streamlit as st
import numpy as np

from PIL import Image
from model.Model import MyModel
from streamlit_image_select import image_select


# Section 1: Heading and Description
"""
# ROSE COLOR DETECTION
Model pendeteksi warna bunga mawar pada input dengan menggunakan metode **Convolutional Neural Network (CNN)** pada Machine Learning. Model ini dibentuk dengan menggunakan model pretrained VGG16 yang sudah dimodifikasi sedemikian rupa agar lebih sederhana dan cepat.
"""


# Section 2: Choosing Image

img = image_select(
    label = st.write("### Pick an Image"),
    images=
    [
        'image/pink-example.jpg',
        'image/red-example.jpg',
        'image/yellow-example.jpg',
    ],  
    captions=['Pink', 'Red', 'Yellow']
)

image_uploaded = st.file_uploader('Chosee an image!', type=['png', 'jpeg', 'jpg'])

st.write("## Model Prediction")


# Section 3: Model Prediction
start_time = time.time()

model = MyModel()   

if image_uploaded is not None:
    st.image(np.array(Image.open(image_uploaded)))
    result = model.predict(image_uploaded)
    st.write("Result:", result)
    end_time = time.time()
    time_consume = end_time - start_time
    st.write("Time consume:", time_consume)
