# Keras FaceNet Pre-Trained Model (88 megabytes)
# https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

import numpy as np
from sklearn.svm import SVC
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

model = load_model('data/model/facenet_keras.h5')


def predictFaces(trainX, trainY, testX, testY):
    trainY, testY = encodeLabels(trainY), encodeLabels(testY)
    trainX, testX = getFacesEmbedding(trainX), getFacesEmbedding(testX)

    svc = SVC(kernel='linear', probability=True)
    svc.fit(trainX, trainY)

    yhatTrain = svc.predict(trainX)
    yhatTest = svc.predict(testX)

    scoreTrain = accuracy_score(trainY, yhatTrain)
    scoreTest = accuracy_score(testY, yhatTest)

    print('Accuracy: train=%.3f, test=%.3f' % (scoreTrain * 100, scoreTest * 100))


def getFacesEmbedding(faces):
    embeddings = []
    for face in faces:
        embedding = getFaceEmbedding(face)
        embeddings.append(embedding)
    embeddings = np.asarray(embeddings)
    return normalizeEmbeddings(embeddings)


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


def encodeLabels(labels):
    return LabelEncoder().fit_transform(labels)
