import cv2
import pytesseract
import numpy as np


# Defining main function
def main(img):
    text = pytesseract.image_to_string(roi)
    # return text
    print(text)


# Read file
with cv2.imread('img01.png') as img:
    img = img
    # img = cv2.imread('./images/img01.png')


# Converting image to grayscale
def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Inverting color
def invert(img):
    return cv2.bitwise_not(img)


# Noise removal
def remove_noise(img):
    return cv2.medianBlur(img, 5)


# Threshold
def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Dilatinf
def dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# Acquiring image dimensions
# height, width, _ = img.shape  # for rgb
# height, width = img.shape  # for grayscale


# Region of interest
def roi(img):
    height, width = img.shape
    roi = img[247:height-90, 585:]
    return roi


# Setting region of interest
# cv2.imshow('img01', img)
# cv2.imshow('roi', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # print(img.shape)


# Calling main function
if __name__ == "__main__":
    main(img)
