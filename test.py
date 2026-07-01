# Load Image and Extract its features
# find the cosine distance of current image with all the 8655 features
# recommend that image to the user

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

detector = MTCNN()

# load image -> face detection

sample_img = cv2.imread('sample/saif dup.png')
result = detector.detect_faces(sample_img)

x, y, width, height = result[0]['box']
face = sample_img[y:y + height, x:x + width]

# cv2.imshow('output', face)
# cv2.waitKey(0)

# extract features from image

image = Image.fromarray(face)
image = cv2.resize(face, (224, 224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')

expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)

result = model.predict(preprocessed_img).flatten()

# print(result)
# print(result.shape)

# find the cosine distance of current image with all the 8655 features


similarity = []

for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    
# print(len(similarity))

# print(list(enumerate(similarity)))

# sort the images on the basis of similarity
# Pick the top image

index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

temp_img = cv2.imread(filenames[index])

cv2.imshow('output', temp_img)
cv2.waitKey(0)