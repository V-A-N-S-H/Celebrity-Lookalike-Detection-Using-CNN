# Suppress warnings to keep the terminal clean
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Bypass SSL verification to allow downloading model weights on Streamlit Cloud
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# NumPy compatibility monkey-patch for older TensorFlow and h5py versions
import numpy as np
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

# Import the actual Keras submodules before we redirect the sys.modules mappings
import sys
import types
try:
    import keras.utils.layer_utils as lu
except ImportError:
    lu = None

try:
    import keras.utils.generic_utils as gu
except ImportError:
    gu = None

# Redirect standalone Keras imports to TensorFlow's Keras to prevent version mismatches
import tensorflow as tf

keras_mappings = {
    'keras': tf.keras,
    'keras.backend': tf.keras.backend,
    'keras.layers': tf.keras.layers,
    'keras.models': tf.keras.models,
}
for old, new in keras_mappings.items():
    sys.modules[old] = new

# Create custom keras.utils to support submodules like layer_utils
keras_utils = types.ModuleType('keras.utils')
for attr in dir(tf.keras.utils):
    setattr(keras_utils, attr, getattr(tf.keras.utils, attr))

if lu is not None:
    keras_utils.layer_utils = lu
    sys.modules['keras.utils.layer_utils'] = lu

sys.modules['keras.utils'] = keras_utils

# Patch old keras-vggface specific internal modules
keras_engine_topology = types.ModuleType('keras.engine.topology')
keras_engine_topology.get_source_inputs = tf.keras.utils.get_source_inputs
sys.modules['keras.engine.topology'] = keras_engine_topology

if gu is not None:
    sys.modules['keras.utils.generic_utils'] = gu
else:
    keras_utils_generic_utils = types.ModuleType('keras.utils.generic_utils')
    keras_utils_generic_utils.get_file = tf.keras.utils.get_file
    sys.modules['keras.utils.generic_utils'] = keras_utils_generic_utils

import cv2
import pickle
from PIL import Image
import streamlit as st
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def feature_extract(img_path, model, detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)

    if len(result) == 0:
        face = img
    else:
        x, y, width, height = result[0]['box']
        face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = cv2.resize(face, (224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index

st.title("Which Bollywood Actor You Look Like?")

st.write("Upload your face to check which Bollywood Actor you look like.")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # save image in the current directory
    if save_uploaded_image(uploaded_image):

        # load the image
        display_image = Image.open(uploaded_image)

        # extract features
        features = feature_extract(os.path.join('uploads', uploaded_image.name), model, detector)

        index_pos = recommend(feature_list, features)
        
        predicted_actor_path = filenames[index_pos].replace('\\', '/')
        predicted_actor_name = os.path.basename(os.path.dirname(predicted_actor_path)).replace('_', ' ')

        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.header("Your Image")
            st.image(display_image, use_column_width=True)
        with col2:
            st.header(f"looks like: {predicted_actor_name}")
            st.image(predicted_actor_path, use_column_width=True)