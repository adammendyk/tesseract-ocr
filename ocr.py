import cv2
import pytesseract
import numpy as np


# Defining main function
def main(img):
    img = get_grayscale(img)
    img = roi(img)
    img = invert(img)
    # img = remove_noise(img, 1)
    # img = thresholding(img)
    # img = dilate(img, 1)
    # img = opening(img, 1)
    show_img(img)
    text = pytesseract.image_to_string(img)
    # return text
    print(text)


# Read file
img = cv2.imread('./images/img01.png')


# Converting image to grayscale
def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Inverting color
def invert(img):
    return cv2.bitwise_not(img)


# Noise removal
def remove_noise(img, size=1):
    return cv2.medianBlur(img, size)


# Threshold
def thresholding(img, vfrom=0, vto=255):
    return cv2.threshold(img, vfrom, vto,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Dilatinf
def dilate(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# Opening - erosion foloved by dilation
def opening(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# Acquiring image dimensions
def get_dimensions(img, rgb=0):
    """rgb defaults to 0 accuming grayscale image"""

    if rgb == 1:
        height, width, _ = img.shape  # for rgb
    else:
        height, width = img.shape  # for grayscale


# Region of interest
def roi(img):
    height, width = img.shape
    roi = img[247:height-90, 585:]
    return roi


# Showing an image
def show_img(img):
    cv2.imshow('img01', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Calling main function
if __name__ == "__main__":
    main(img)
