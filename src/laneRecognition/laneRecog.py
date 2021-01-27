import os
import re
import cv2
import numpy as np

import matplotlib.pyplot as plt

theta = np.pi / 180


def interested_region(img):
    imshape = img.shape
    lower_left = [imshape[1] / 9, imshape[0]]
    lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    polygons = np.array([lower_left, top_left, top_right, lower_right])
    if len(img.shape) > 2:
        mask_color_ignore = 255 * img.shape[2]
    else:
        mask_color_ignore = 255
    cv2.fillPoly(np.zeros_like(img), polygons, mask_color_ignore)
    return cv2.bitwise_and(img, np.zeros_like(img))




def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def canny_edge_detector(image):
    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


cap = cv2.VideoCapture("datasets\test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny_edge_detector(frame)
    cropped_image = interested_region(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                            np.array([]), minLineLength=40,
                            maxLineGap=5)

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = draw_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("results", combo_image)

    # When the below two will be true and will press the 'q' on
    # our keyboard, we will break out from the loop

    # # wait 0 will wait for infinitely between each frames.
    # 1ms will wait for the specified time only between each frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the video file
cap.release()

# destroy all the windows that is currently on
cv2.destroyAllWindows()