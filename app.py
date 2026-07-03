# Suppress warnings to keep the terminal clean
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        
        predicted_actor_path = filenames[index_pos]
        predicted_actor_name = os.path.basename(os.path.dirname(predicted_actor_path)).replace('_', ' ')

        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.header("Your Image")
            st.image(display_image, width=300)
        with col2:
            st.header(f"looks like: {predicted_actor_name}")
            st.image(predicted_actor_path, width=600)