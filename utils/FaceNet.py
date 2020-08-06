# Keras FaceNet Pre-Trained Model (88 megabytes)
# https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

import numpy as np
from utils import OpenCV
from keras.models import load_model
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer

recognitionRate = 0.3
model = load_model('data/model/facenet_keras.h5')


def searchFaces(targets, people, photos):
    peopleEmbedding = getFacesEmbedding(people)
    targetsEmbedding = getFacesEmbedding(targets)

    for personEmbedding in peopleEmbedding:
        name = 'unknown'
        minDistance = float('inf')

        for targetEmbedding in targetsEmbedding:

            distancePeopleTarget = cosine(targetEmbedding['face'], personEmbedding['face'])

            if distancePeopleTarget < recognitionRate and distancePeopleTarget < minDistance:
                name = targetEmbedding['name']
                minDistance = distancePeopleTarget

        image, axisX, axisY = photos[personEmbedding['filename']], personEmbedding['axisX'], personEmbedding['axisY']

        if name == 'unknown':
            OpenCV.showRectangle(image, axisX, axisY, OpenCV.colorFailure)
            OpenCV.showText(image, name, axisX, OpenCV.colorFailure)
        else:
            OpenCV.showRectangle(image, axisX, axisY, OpenCV.colorSuccess)
            OpenCV.showText(image, name + f" {minDistance:.2f}", axisX, OpenCV.colorSuccess)

    return photos


def getFacesEmbedding(facesDict):
    embeddings = []
    for faceDict in facesDict:
        embedding = getFaceEmbedding(faceDict['face'])
        embeddings.append(embedding)

    embeddings = np.asarray(embeddings)
    embeddings = normalizeEmbeddings(embeddings)

    for i in range(len(embeddings)):
        facesDict[i]['face'] = embeddings[i]

    return facesDict


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


def normalizeEmbeddings(embeddings):
    encoder = Normalizer(norm='l2')
    return encoder.transform(embeddings)
