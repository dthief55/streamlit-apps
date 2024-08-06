import numpy as np
import tensorflow as tf
from PIL import Image

class MyModel:
    label_names = ['Pink', 'Red', 'Yellow']
    
    def __init__(self):
        self.model = tf.keras.models.load_model('model/rose_color_classification.h5')

    def preprocessing(self, image):
        image = Image.open(image)
        image = image.resize((224, 224))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)
        
        return image
        
    def predict(self, image):
        image = self.preprocessing(image)
        predictions_vector = self.model.predict(image)
        predicted_idx = [self.label_names[np.argmax(prediction)] for prediction in predictions_vector]
        label = predicted_idx[0]
        
        return label