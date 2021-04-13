import os
import re
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt

theta = np.pi / 180


def interested_region(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    for line in lines:
        if len(line) == 4:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) > 800:
                x2 = int(x1 + (x2 - x1) / 2)
                coord =[[x1,y1],[x1/1.2,y1/1.1],[int((x1+x2)/2.8),int((y1+y2)/2.5)],[x2/1.1,y2],[int((x1+x2)/2.6),int((y1+y2)/2.8)]]
                print(coord)
                coord=np.array(coord,np.int32)
                coord=coord.reshape((-1,1,2))
                isClosed=False
                img=cv2.polylines(img,[coord],isClosed,color,thickness)
            else:
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
    global left_line, right_line
    left_fit = []
    right_fit = []
    left_line = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
            left_fit_average = np.average(left_fit, axis=0)
            left_line = create_coordinates(image, left_fit_average)
        else:
            right_fit.append((slope, intercept))
            right_fit_average = np.average(right_fit, axis=0)
            right_line = create_coordinates(image, right_fit_average)
    if len(left_line) == 0 :
        left_line[0] = right_line[0] + 300
        left_line[1] = right_line[1]
        left_line[2] = right_line[2] + 300
        left_line[3] = right_line[3]
        print("completer")
    return np.array([left_line, right_line])


cap = cv2.VideoCapture("test_video/challenge.mp4")
i = 0
while cap.isOpened():
    try:
        _, frame = cap.read()
        if frame is None:
            print("end of video file")
            break
        combo_image = frame
        if i % 24 == 0:
            canny_image = canny_edge_detector(frame)
            cropped_image = -interested_region(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                                    np.array([]), minLineLength=40,
                                    maxLineGap=5)

            averaged_lines = average_slope_intercept(frame, lines)
        line_image = draw_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("results", combo_image)
        i += 1
        time.sleep(1)
    except:
        print("Image quality does not allow line recognition")
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