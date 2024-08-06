import time
import streamlit as st
import numpy as np

from PIL import Image
from model.Model import MyModel
from streamlit_image_select import image_select


# Section 1: Heading and Description



# Section 2: Choosing Image

img = image_select(
    label = st.write("## Pick an Image"),
    images=
    [
        'image/Pink/al-soot--pldRipVgx4-unsplash.jpg',
        'image/Red/Red Rose 1.jpg',
        'image/Yellow/Yellow Rose (1).jpg'
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