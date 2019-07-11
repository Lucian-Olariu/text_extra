import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    from cv2 import cv2
except ImportError:
    pass


def threshold_slow(T, image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = image[y, x] * T

    # return the thresholded image
    return image

cwd = os.path.dirname(os.path.abspath(__file__))
input_dir_path = cwd + '/data/test_input'
output_dir_path = cwd + '/data/test_output'

input_image_names = os.listdir(input_dir_path)
print(input_image_names)


img = cv2.imread(input_dir_path + '/' + input_image_names[1])
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bin, thresh = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)

kernel = np.ones((17,17),np.uint8)
dilation = cv2.erode(thresh,kernel,iterations = 1)


result = np.concatenate((thresh, dilation), axis=1)

plt.imshow(result, cmap="gray")
plt.show()
