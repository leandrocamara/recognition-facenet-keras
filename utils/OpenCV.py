import cv2
from PIL import Image

colorFailure, colorSuccess = (255, 0, 0), (0, 255, 0)


def getImage(path):
    image = cv2.imread(path)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imageRGB


def resizeImage(image, size):
    return cv2.resize(image, size)


def resizeScaleImage(image, scale):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def showRectangle(image, x, y, color=(255, 255, 255)):
    cv2.rectangle(image, x, y, color, 2)


def showText(image, text, x, color=(255, 255, 255)):
    cv2.putText(image, text, x, cv2.FONT_HERSHEY_PLAIN, 1.5, color, 1)


def rotate(image, angle):
    return Image.fromarray(image).rotate(angle)


def saveFile(url, image):
    Image.fromarray(image).save(url)
