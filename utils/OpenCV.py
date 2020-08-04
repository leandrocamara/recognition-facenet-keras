import cv2


def getImage(path):
    image = cv2.imread(path)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imageRGB


def resizeImage(image, size):
    return cv2.resize(image, size)


def showRectangle(image, x, y):
    cv2.rectangle(image, x, y, (255, 255, 255), 2)
