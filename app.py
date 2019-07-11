import os
import io
import numpy as np

from PIL import Image

try:
    from cv2 import cv2
except ImportError:
    pass

import json

cwd = os.path.dirname(os.path.abspath(__file__))
input_dir_path = cwd + '/data/test_input'
output_dir_path = cwd + '/data/test_output'
input_image_names = os.listdir(input_dir_path)


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


wnd = 'Sandbox'
cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)

cv2.createTrackbar("Threshold", wnd, 172, 255, nothing)
cv2.createTrackbar("Blur", wnd, 17, 255, nothing)
cv2.createTrackbar("Rotation", wnd, 145, 360, nothing)
cv2.createTrackbar("Clip_limit", wnd, 76, 255, nothing)
cv2.createTrackbar("Tile_grid_size", wnd, 11, 255, nothing)


img = cv2.imread(input_dir_path + '/' + input_image_names[1])


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=7.6, tileGridSize=(11, 11))
cl1 = clahe.apply(gray_img)

blurr = cv2.GaussianBlur(cl1, (17, 17), 0)

circles = cv2.HoughCircles(blurr, cv2.HOUGH_GRADIENT, 1.2, 170, 70, 80)

radius = 0
rect_height = 0
rect_width = 0

if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the test_output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)

        cv2.rectangle(img, (x-r, y-r), (x+r, y+r), (0, 128, 255), 2)
        rect_height = y
        rect_width = x
        radius = r

top = rect_width - radius
left = rect_height - radius
bottom = rect_width + radius
right = rect_height + radius
roi = gray_img[left:right, top:bottom]

while (1):
    thresh_bin_val = cv2.getTrackbarPos("Threshold", wnd)
    rotation = cv2.getTrackbarPos("Rotation", wnd)
    blur_kernel_value = (makeValueOdd(cv2.getTrackbarPos("Blur", wnd)), makeValueOdd(cv2.getTrackbarPos("Blur", wnd)))
    clip_limit = cv2.getTrackbarPos("Clip_limit", wnd)
    tile_grid_size = (cv2.getTrackbarPos("Tile_grid_size", wnd), cv2.getTrackbarPos("Tile_grid_size", wnd))

    rows = roi.shape[0]
    cols = roi.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    output = cv2.warpAffine(roi, M, (cols, rows))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    output = clahe.apply(output)

    output = cv2.GaussianBlur(output, blur_kernel_value, 0)


    ret, output = cv2.threshold(output, thresh_bin_val, 255, cv2.THRESH_BINARY)


    cv2.imshow(wnd, ~output)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        print("Extracting...")
        text_list = detect_text(output)
        if len(text_list) is not 0:
            text_list.pop(0)
            text_string = ''.join(text_list)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            cv2.putText(output, text_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(output_dir_path + '/test_2.png', ~output)

            config = {
                "Threshold_value": thresh_bin_val,
                "Rotation": rotation,
                "Blur_kernel_value": blur_kernel_value,
                "Clip_limit": clip_limit,
                "Tile_grid_size": tile_grid_size
            }

            with open(cwd + '/config.json', 'w') as file:
                file.write(json.dumps(config))

        else:
            text_string = 'Could not extract text'
            cv2.putText(output, text_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_dir_path + '/test_2.png', ~output)

            config = {
                "Threshold_value": thresh_bin_val,
                "Rotation": rotation,
                "Blur_kernel_value": blur_kernel_value,
                "Clip_limit": clip_limit,
                "Tile_grid_size": tile_grid_size
            }

            with open(cwd + '/config.json', 'w') as file:
                file.write(json.dumps(config))

    elif k == 27:
        break

cv2.destroyAllWindows()
