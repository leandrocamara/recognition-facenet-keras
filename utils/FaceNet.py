# Keras FaceNet Pre-Trained Model (88 megabytes)
# https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

import numpy as np
from keras.models import load_model

model = load_model('data/model/facenet_keras.h5')


def getFacesEmbedding(faces):
    embeddings = []
    for face in faces:
        embedding = getFaceEmbedding(face)
        embeddings.append(embedding)
    return np.asarray(embeddings)


def getFaceEmbedding(face):
    face = standardizeFace(face)
    sample = transformFaceToSample(face)
    yhat = model.predict(sample)
    return yhat[0]


def standardizeFace(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    return (face - mean) / std


def transformFaceToSample(face):
    return np.expand_dims(face, axis=0)
