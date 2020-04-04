"""Utilities module for Image Classifier project"""

import datetime
import tensorflow as tf

def datestr():
    "Returns date and time string in a particular format"
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

def process_image(image):
    "Take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape (224, 224, 3)."
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224,224))
    image /= 255
    return image.numpy()
