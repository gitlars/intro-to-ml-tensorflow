"""Predict the top flower names from an image along with their corresponding probabilities.

Basic usage:
$ python predict.py /path/to/image saved_model

Example:
$ python predict.py test_images/cautleya_spicata.jpg 2020-04-04_20-48-19Z.h5

Options:
    --top_k : Return the top KKK most likely classes:
    $ python predict.py /path/to/image saved_model --top_k KKK

    --category_names : Path to a JSON file mapping labels to flower names:
    $ python predict.py /path/to/image saved_model --category_names map.json

For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

Basic usage:
$ python predict.py ./test_images/orchid.jpg my_model.h5

Options:
    Return the top 3 most likely classes:
    $ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3

    Use a label_map.json file to map labels to flower names:
    $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
"""

from util import datestr, process_image
import argparse
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_hub import KerasLayer

# Parse input arguments
parser = argparse.ArgumentParser(
    description='Predict flower name and probability from an image'
)

parser.add_argument('image_path', type=str, help='path to input image file')
parser.add_argument('model_path', type=str, help='model file in HDF5 format')
parser.add_argument('--top_k', type=int, required=False, nargs='?', default=5, metavar='K', help='number of top results to return (default=%(default)s)')
parser.add_argument('--category_names', type=str, required=False, nargs='?', default='label_map.json', metavar='JSON_FILEPATH', help='path to label-to-name JSON file (default=\'%(default)s\')')

args = parser.parse_args()

# Show TensorFlow version information
#print('Using:')
#print('\t\u2022 TensorFlow version:', tf.__version__)
#print('\t\u2022 tf.keras version:', tf.keras.__version__)
#print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# Read label mapping
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

# Read and prepare the input image
im = Image.open(args.image_path)
im = np.asarray(im)
im = process_image(im)
im = np.expand_dims(im, axis=0)

# Load the model
model = tf.keras.models.load_model(
    args.model_path, custom_objects={'KerasLayer':KerasLayer}
)

# Predict the flower category
topk = args.top_k
probs = model.predict(im)
sort_indices = np.argsort(probs[0])
sort_flip_indices = np.flip(sort_indices, axis=0)
topk_indices = sort_flip_indices[:topk]
topk_probs = probs[0][topk_indices]

topk_keys = ["{}".format(x + 1) for x in topk_indices]
topk_names = [class_names[x] for x in topk_keys]

# Output the results
print("\nFor input image {}, model {} has the following top {} predictions:".format(
    args.image_path, args.model_path, topk
))

for i in np.arange(topk):
    print("   #{}: {} (key = {}, probability = {:.6f})".format(
        i + 1, topk_names[i], topk_keys[i], topk_probs[i]
    ))
