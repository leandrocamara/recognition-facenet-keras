import warnings
import numpy as np
from utils import OpenCV
from mtcnn.mtcnn import MTCNN

warnings.simplefilter('ignore')


def extractFaces(path, requiredSize=(160, 160)):
    faces = []
    mtcnn = MTCNN()
    image = OpenCV.getImage(path)

    results = mtcnn.detect_faces(image)

    for result in results:
        if result['confidence'] < 0.97:
            continue

        face, x, y = getFace(image, result['box'])
        face = OpenCV.resizeImage(face, requiredSize)
        faces.append(face)

        OpenCV.showRectangle(image, x, y)

    return image, faces  # np.asarray(image)


def getFace(pixels, box):
    x1, y1, w, h = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    face = pixels[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
