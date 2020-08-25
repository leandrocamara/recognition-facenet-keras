import numpy as np
from os import listdir
from utils import OpenCV
from mtcnn.mtcnn import MTCNN

import cv2

mtcnn = MTCNN()


def extractFaces(path, filename, requiredSize=(160, 160)):
    faces = []
    image = OpenCV.getImage(path)

    results = mtcnn.detect_faces(image)

    for result in results:
        if result['confidence'] < 0.97:
            continue

        face, x, y = getFace(image, result['box'])
        leftEye, rightEye = getEyes(x, result['keypoints'])

        # Test Align Face
        cv2.circle(face, leftEye, 2, (0, 155, 255), 2)
        cv2.circle(face, rightEye, 2, (0, 155, 255), 2)
        cv2.line(face, leftEye, rightEye, (67, 67, 67), 2)

        face = OpenCV.resizeImage(face, requiredSize)
        faces.append({'face': face, 'axisX': x, 'axisY': y, 'filename': filename})

    return faces, image


def loadInputFaces(directory):
    allFaces, photos = [], dict()
    for filename in listdir(directory):
        path = directory + '/' + filename
        faces, image = extractFaces(path, filename)
        allFaces.extend(faces)
        photos[filename] = image

    for i in range(len(allFaces)):
        allFaces[i]['name'] = 'person_' + str(i + 1)

    return np.asarray(allFaces), photos


def getFace(pixels, box):
    x1, y1, w, h = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    face = pixels[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def getEyes(faceX, keyPoints):
    left = keyPoints['left_eye']
    right = keyPoints['right_eye']
    return (left[0] - faceX[0], left[1] - faceX[1]), (right[0] - faceX[0], right[1] - faceX[1])
