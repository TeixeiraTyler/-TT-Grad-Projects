import cv2
import numpy as np
from matplotlib import pyplot as plt

input_image1 = cv2.imread('image1.jpg', 0)
input_image2 = cv2.imread('image2.jpg', 0)
input_image3 = cv2.imread('image3.jpg', 0)

current_Image = input_image1

# Show original image before any thresholding.
cv2.imshow('Original Image', current_Image)
cv2.waitKey(0)

# Manual thresholding that user must manipulate to change boundary for binary selection.
ret, threshold_1 = cv2.threshold(current_Image, 127, 255, cv2.THRESH_BINARY)

# Automatic thresholding that uses openCV otsu thresholding.
ret, threshold_2 = cv2.threshold(current_Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Choose which thresholding to show.
current_Thresholding = threshold_2

# Show image after thresholding.
cv2.imshow('Threshold Binary', current_Thresholding)
cv2.waitKey(0)

# Show histogram of image after thresholding.

plt.hist(threshold_1.ravel(), 256, [0, 256])
plt.show()
