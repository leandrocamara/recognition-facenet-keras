import math
from os import listdir

import numpy as np
from mtcnn.mtcnn import MTCNN

from utils import OpenCV

mtcnn = MTCNN()


def extractFaces(path, filename, requiredSize=(160, 160)):
    faces = []
    image = OpenCV.getImage(path)
    scale = 2000  # 1000

    axisMax = max(image.shape)
    image = OpenCV.resizeScaleImage(image, scale=(scale / axisMax))

    results = mtcnn.detect_faces(image)

    for result in results:
        if result['confidence'] < 0.98:
            continue

        face, axisX, axisY = getFace(image, result['box'])
        face = alignFace(face, axisX, result['keypoints'])
        face = OpenCV.resizeImage(face, requiredSize)
        faces.append({'face': face, 'axisX': axisX, 'axisY': axisY, 'filename': filename})

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


def alignFace(face, axisX, keypoints):
    leftEye, rightEye = getEyes(axisX, keypoints)

    pointRightAngle = (leftEye[0], rightEye[1])  # leftEyeX, rightEyeY
    direction = 1  # rotate inverse direction of clock

    if leftEye[1] > rightEye[1]:  # leftEyeY > rightEyeY
        pointRightAngle = (rightEye[0], leftEye[1])  # rightEyeX, leftEyeY
        direction = -1  # rotate same direction to clock

    # cv2.circle(face, leftEye, 2, (0, 155, 255), 2)
    # cv2.circle(face, rightEye, 2, (0, 155, 255), 2)
    # cv2.circle(face, pointRightAngle, 2, (0, 155, 255), 2)

    # cv2.line(face, leftEye, rightEye, (67, 67, 67), 2)
    # cv2.line(face, leftEye, pointRightAngle, (67, 67, 67), 2)
    # cv2.line(face, rightEye, pointRightAngle, (67, 67, 67), 2)

    angleEyes = getAngleEyes(leftEye, rightEye, pointRightAngle)

    if direction == -1:
        angleEyes = 90 - angleEyes

    return np.array(OpenCV.rotate(face, direction * angleEyes))


def getAngleEyes(leftEye, rightEye, pointRightAngle):
    b = euclideanDistance(rightEye, leftEye)
    a = euclideanDistance(leftEye, pointRightAngle)
    c = euclideanDistance(rightEye, pointRightAngle)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    angle = np.arccos(cos_a)
    return (angle * 180) / math.pi


def euclideanDistance(a, b):
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))
