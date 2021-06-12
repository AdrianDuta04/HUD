import cv2
from PIL import Image
from tensorflow import keras
import numpy as np
import random
import time

def filter_regions(rects):
    x_array = []
    new_rects = []
    for i in range(len(rects)):
        if rects[i][0] not in x_array and rects[i][0] > 0 and rects[i][1] > 0 and \
                1300 < rects[i][2] * rects[i][3] < 2300 and \
                (rects[i][2] == rects[i][3] or rects[i][3] - rects[i][3] / 8 <= rects[i][2] <= rects[i][3] + rects[i][
                    3] / 8):
            new_rects.append(rects[i])
            x_array.append(rects[i][0])
    return new_rects


def searchSignQaulity(input):
    image = cv2.imread(input)
    new = image[image.shape[0]:(int(image.shape[0] / 9)):-1, image.shape[1]:int(image.shape[1] / 2):-1]
    new = new[image.shape[0]:int(image.shape[0] / 4):-1, image.shape[1]:0:-1]
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(new)
    # ss.switchToSelectiveSearchFast()
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    rects = filter_regions(rects)
    for (x, y, w, h) in rects:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(new, (x, y), (x + w, y + h), color, 2)
    cv2.imshow("Output", new)
    key = cv2.waitKey(0) & 0xFF
    return new


def searchSignFast(input):
    image = cv2.imread(input)
    new = image[image.shape[0]:(int(image.shape[0] / 9)):-1, image.shape[1]:int(image.shape[1] / 2):-1]
    new = new[image.shape[0]:int(image.shape[0] / 4):-1, image.shape[1]:0:-1]
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(new)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    rects = filter_regions(rects)
    for (x, y, w, h) in rects:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(new, (x, y), (x + w, y + h), color, 2)
    cv2.imshow("Output", new)
    key = cv2.waitKey(0) & 0xFF
    return new


def load_model():
    global model
    model = keras.models.load_model("trained_models/sign_recognition_model_last.h5")
    image_dummy = Image.open("../../data/test/00000.png")
    image_dummy = image_dummy.resize((32, 32))
    image_dummy = np.expand_dims(image_dummy, axis=0)
    image_dummy = np.array(image_dummy)
    pred_dummy = model.predict_classes([image_dummy])[0]
    sign = classes[pred_dummy + 1]
    print(sign)


def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred + 1]
    return sign, pred


def interested_region(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    for line in lines:
        if len(line) == 4:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) > 800:
                x2 = int(x1 + (x2 - x1) / 2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def canny_edge_detector(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    if len(left_line) == 0:
        left_line[0] = right_line[0] + 300
        left_line[1] = right_line[1]
        left_line[2] = right_line[2] + 300
        left_line[3] = right_line[3]
    if left_line[0] < 0:
        left_line[0] = 0
    if left_line[2] < 0:
        left_line[2] = left_line[3]
    print(left_line)
    return np.array([left_line, right_line])


def laneR(title):
    cap = cv2.VideoCapture(title)
    i = 0
    while cap.isOpened():
        try:
            _, frame = cap.read()
            copy_frame = frame
            frame = cv2.addWeighted(frame, 1, frame, 0.1, 2)
            if frame is None:
                print("end of video file")
                break
            combo_image = frame
            if i % 12 == 0:
                canny_image = canny_edge_detector(frame)
                cropped_image = -interested_region(canny_image)
                lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50,
                                        np.array([]), minLineLength=20,
                                        maxLineGap=3)

                averaged_lines = average_slope_intercept(frame, lines)
            line_image = draw_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(copy_frame, 0.1, line_image, 1, 2)
            # time.sleep(0.1)
            cv2.imshow("results", combo_image)
            i += 1
        except:
            print("Image quality does not allow line recognition")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()