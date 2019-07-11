import os
import io
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass

import json
from PIL import Image
import time

cwd = os.path.dirname(os.path.abspath(__file__))
input_dir_path = cwd + '/data/unprocessed'
output_dir_path = cwd + '/data/processed'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cwd + '/data/credentials/vision/credentials.json'

with open(cwd + '/config.json', 'r') as file:
    config = file.read()

config = json.loads(config)

def nothing(x):
    pass


def makeValueOdd(value):
    if (value % 2) is not 0:
        return value
    return value + 1

def detect_text(imageArray):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    content = Image.fromarray(imageArray)

    imgByteArr = io.BytesIO()
    content.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    image = vision.types.Image(content=imgByteArr)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    text_list = []
    for text in texts:
        print('\n"{}"'.format(text.description))
        text_list.append(text.description)
        # vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        #
        # print('bounds: {}'.format(','.join(vertices)))

    return text_list


thresh_bin_val = config['Threshold_value']
rotation = config["Rotation"]
blur_kernel_value = config["Blur_kernel_value"]
clip_limit = config["Clip_limit"]
tile_grid_size = config["Tile_grid_size"]

while(1):
    image_names = os.listdir(input_dir_path)
    if len(image_names) > 0:
        print("Found {} new images in input folder...".format(len(image_names)))

        for image_name in image_names:
            img = cv2.imread(input_dir_path + '/' + image_name)
            print('Processing image {}'.format(image_name))
            print('Extracting region of interest...')
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=7.6, tileGridSize=(11, 11))
            cl1 = clahe.apply(gray_img)
            blurr = cv2.GaussianBlur(cl1, (17, 17), 0)

            circles = cv2.HoughCircles(blurr, cv2.HOUGH_GRADIENT, 1.2, 170, 70, 80)

            radius = 0
            rect_height = 0
            rect_width = 0

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for (x, y, r) in circles:
                    rect_height = y
                    rect_width = x
                    radius = r

            top = rect_width - radius
            left = rect_height - radius
            bottom = rect_width + radius
            right = rect_height + radius
            roi = gray_img[left:right, top:bottom]

            print('Rotating image to config parameter value')
            rows = roi.shape[0]
            cols = roi.shape[1]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
            output = cv2.warpAffine(roi, M, (cols, rows))

            print("Applying Contrast Limited Adaptive Histogram Equalization")
            print(tuple(tile_grid_size))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile for tile in tile_grid_size))
            output = clahe.apply(output)

            print("Applying Gausian Blurr")
            output = cv2.GaussianBlur(output, tuple(blur_kernel_value), 0)

            print("Thresholding")
            ret, output = cv2.threshold(output, thresh_bin_val, 255, cv2.THRESH_BINARY)

            print("Proceeding to text extraction, plrease stand by...")

            text_list = detect_text(~output)
            text_string = "No text found"
            if len(text_list) is not 0:
                text_list.pop(0)
                text_string = ''.join(text_list)
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                cv2.putText(output, text_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite(output_dir_path + '/' + str(time.time()) + '_' + image_name , ~output)
            else:
                cv2.putText(output, text_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(output_dir_path + '/' + str(time.time()) + '_' + image_name , ~output)

            print("{} was processed. Removing...".format(image_name))
            os.remove(input_dir_path + '/' + image_name)
    else:
        print("No new images found. Awaiting images...")

    time.sleep(1)