import sys
import json
import threading

from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QFileDialog, QLabel, QDialog, \
    QGridLayout, QCheckBox, QStyle, QVBoxLayout, QHBoxLayout, QStatusBar, QSlider, QWidget
from PyQt5.uic.properties import QtGui
import cv2
from PIL import Image
from tensorflow import keras
import numpy as np
import random
import time
import imutils
import pytesseract
import _thread


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def hasLetters(inputString):
    return any(char in 'QWERTYUIOPASDFGHJKLZXCVBNM' for char in inputString)


def hasSpecialChars(inputString):
    return any(char in '[]()*!@#$%^&{}:;""<>,.~`_|\\+=' for char in inputString)


def hasLittleChars(inputString):
    return any(char in 'qwertyuiopasdfghjklzxcvbnm' for char in inputString)


def startWrong(inputString):
    return inputString[0] in '-'


def license_plate(image, license_list):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(image, image, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        text = text.strip()
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace(' ', '')
        text = text.strip('\n')
        if checkValidLicense(license_list, text):
            license_list.append(text)
            print(license_list)
            print("Detected Number is:", text)
            cv2.imshow('license',Cropped)
            key = cv2.waitKey(0) & 0xFF


def checkValidLicense(license_list, text):
    return 5 < len(text) < 10 and hasNumbers(text) and hasLetters(text) and not hasSpecialChars(
        text) and not hasLittleChars(text) and not startWrong(text) and (text not in license_list)


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


def searchSignQaulity(image, arr):
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


def searchSignFast(image, arr):
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
        arr.append(new[y:(y + h), x:(x + w)])
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


def classify(image):
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])
    pred_classes = model.predict_classes([image])
    prediction_scor = np.max(pred[0])
    sign = classes[pred_classes[0] + 1]
    return sign, pred_classes[0]


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


def color_seg(img):
    blur = cv2.blur(img, (5, 5))
    blur0 = cv2.medianBlur(blur, 5)
    blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
    blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)
    hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    low_red1 = np.array([0, 70, 60])
    high_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, low_red1, high_red1)
    low_red2 = np.array([170, 70, 60])
    high_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, low_red2, high_red2)
    low_blue = np.array([94, 127, 20])
    high_blue = np.array([126, 255, 200])
    mask3 = cv2.inRange(hsv, low_blue, high_blue)
    light_white = (0, 20, 80)
    dark_white = (220, 220, 220)
    mask_white = cv2.inRange(hsv, light_white, dark_white)
    final_mask = mask1 + mask_white + mask2 + mask3
    final_result = cv2.bitwise_and(img, img, mask=final_mask)
    return final_result


def signThread(cap, ThreadId, signsArray):
    i = 0
    while cap.isOpened():
        _, frame = cap.read()
        image_for_sign = frame
        if i % 30 == 0:
            final_result = color_seg(image_for_sign)
            arr = []
            new_image = searchSignFast(final_result, arr)
            for j in arr:
                image = Image.fromarray(j)
                sign, pred = classify(image)
                if pred not in signsArray:
                    signsArray.append(pred)


def laneThread(cap, threadId):
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
            image_for_sign = frame
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


def laneSign(videoIni):
    video = cv2.VideoCapture(videoIni)
    signsArray = []
    video2 = cv2.VideoCapture(videoIni)
    try:
        t1 = threading.Thread(target=signThread, args=(video2, 1, signsArray))
        t2 = threading.Thread(target=laneThread, args=(video, 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print(signsArray)
    except:
        print("Error: unable to start thread")


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


def license_place_video(cap):
    license_list = []
    while cap.isOpened():
        _, frame = cap.read()
        license_plate(frame, license_list)
        cv2.waitKey(0)


def accidentPlate(list_of_frames):
    license_list = []
    for frame in list_of_frames:
        license_plate(frame, license_list)


classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}


class MainDialog(QMainWindow):
    def __init__(self, main, parent=None):
        super(MainDialog, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 1000
        self.main = main
        self.setWindowTitle("Main functionality display")

        self.text = self.createLabel(60, 20, 'HEAD UP DISPLAY')
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.button3 = self.createButton(130, 60, 100, 50, 'Accident')
        self.button3.clicked.connect(self.pericol)
        self.button3.hide()
        self.button1 = self.createButton(240, 60, 100, 30, 'Incarcare fisier')
        self.button1.clicked.connect(self.uploadFile)
        self.button2 = self.createButton(200, 120, 100, 30, 'Process')
        self.button2.clicked.connect(self.combine)
        self.button2.hide()
        self.acc = 0

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def accident(self):
        license_place_video(cv2.VideoCapture(self.video))

    def pericol(self):
        self.acc = 1

    def process(self):
        laneSign(self.video)

    def uploadFile(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.video = file_name[38:]
        self.video = ".." + self.video
        self.textbox.setText(file_name)
        self.button2.show()
        self.button3.show()

    def doNothig(self):
        self.close()

    def combine(self):
        cap = cv2.VideoCapture(self.video)
        i = 0
        signsArray = []
        accidentScene = []
        while cap.isOpened():
            try:
                if self.acc == 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    accidentPlate(accidentScene)
                    break
                _, frame = cap.read()
                if len(accidentScene) == 60:
                    accidentScene.pop()
                accidentScene.append(frame)
                copy_frame = frame
                frame = cv2.addWeighted(frame, 1, frame, 0.1, 2)
                if frame is None:
                    print("end of video file")
                    break
                combo_image = frame
                image_for_sign = frame
                if i % 12 == 0:
                    canny_image = canny_edge_detector(frame)
                    cropped_image = -interested_region(canny_image)
                    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50,
                                            np.array([]), minLineLength=20,
                                            maxLineGap=3)

                    averaged_lines = average_slope_intercept(frame, lines)
                image_for_sign = frame
                if i % 30 == 0:
                    final_result = color_seg(image_for_sign)
                    arr = []
                    new_image = searchSignFast(final_result, arr)
                    for j in arr:
                        image = Image.fromarray(j)
                        sign, pred = classify(image)
                        if pred not in signsArray:
                            signsArray.append(pred)
                line_image = draw_lines(frame, averaged_lines)
                combo_image = cv2.addWeighted(copy_frame, 0.1, line_image, 1, 2)
                # time.sleep(0.1)
                cv2.imshow("results", combo_image)
                i += 1
            except:
                print("Image quality does not allow line recognition")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        for i in signsArray:
            image2 = cv2.imread(
                '/home/adrian/Documents/licenta/HUD/src/signRecognition/signs/' + str(i + 1) + '.png')
            image2 = cv2.resize(image2, (160, 160))
            cv2.imshow("Rectangle ", image2)
            key = cv2.waitKey(0) & 0xFF
        cap.release()
        cv2.destroyAllWindows()


class SearchSign(QMainWindow):
    def __init__(self, parent=None):
        super(SearchSign, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 1000
        self.dialogs = list()
        self.imageFile = ""
        self.setWindowTitle("Sign Search")
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Creare textBox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 30)

        # Creare buton de incarcare fisier
        self.button1 = self.createButton(300, 20, 100, 30, 'Incarcare imagine')
        self.button1.clicked.connect(self.uploadFile2)

        self.button2 = self.createButton(550, 120, 100, 30, 'Fast')
        self.button2.clicked.connect(self.searchFast)
        self.button2.hide()

        self.button3 = self.createButton(550, 150, 100, 30, 'Qaulity')
        self.button3.clicked.connect(self.searchQuality)
        self.button3.hide()

        self.button4 = self.createButton(550, 180, 160, 30, 'Image segmenation fast')
        self.button4.clicked.connect(self.segmentation_fast)
        self.button4.hide()

        self.button5 = self.createButton(550, 210, 160, 30, 'Image segmenation quality')
        self.button5.clicked.connect(self.segmentation_quality)
        self.button5.hide()

        self.video = ""
        self.label2 = QLabel(self)
        self.label2.move(20, 90)
        self.lb = QLabel(self)
        self.lbPredict = QLabel(self)

    def searchFast(self):
        image = cv2.imread(self.imageFile)
        arr = []
        new_image = searchSignFast(image, arr)

    def segmentation_fast(self):
        img = cv2.imread(self.imageFile)
        final_result = color_seg(img)
        arr = []
        new_image = searchSignFast(final_result, arr)
        for i in arr:
            image = Image.fromarray(i)
            sign, pred = classify(image)
            image2 = cv2.imread(
                '/home/adrian/Documents/licenta/HUD/src/signRecognition/signs/' + str(pred + 1) + '.png')
            image2 = cv2.resize(image2, (160, 160))
            cv2.imshow("Rectangle ", image2)

    def segmentation_quality(self):
        img = cv2.imread(self.imageFile)
        final_result = color_seg(img)
        searchSignQaulity(final_result)

    def searchQuality(self):
        image = cv2.imread(self.imageFile)
        searchSignQaulity(image)

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def doNothig(self):
        self.close()

    def uploadFile2(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        name = file_name.split('/')
        self.textbox.setText(file_name)
        self.label2.setText(name[-1])
        self.label2.adjustSize()
        self.imageFile = file_name
        pixmap = QPixmap(file_name)
        self.lb.resize(520, 520)
        self.lb.move(20, 120)
        self.lb.setPixmap(pixmap.scaled(self.lb.size(), Qt.IgnoreAspectRatio))
        self.button2.show()
        self.button3.show()
        self.button4.show()
        self.button5.show()


class LaneMain(QMainWindow):
    def __init__(self, parent=None):
        super(LaneMain, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
        self.dialogs = list()
        self.imageFile = ""
        self.setWindowTitle("Lane Recognition")
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Creare textBox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 30)

        # Creare buton de incarcare fisier
        self.button1 = self.createButton(300, 20, 100, 30, 'Incarcare fisier')
        self.button1.clicked.connect(self.uploadFile)
        self.button2 = self.createButton(200, 120, 100, 30, 'Process')
        self.button2.clicked.connect(self.laneReco)
        self.button2.hide()
        self.video = ""
        self.label2 = QLabel(self)
        self.label2.move(20, 90)
        self.button5 = self.createButton(300, 300, 200, 50, 'Exit')
        self.button5.clicked.connect(self.doNothig)

    def laneReco(self):
        laneR(self.video)

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def doNothig(self):
        self.close()

    def uploadFile(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.video = file_name[38:]
        self.video = ".." + self.video
        name = file_name.split('/')
        self.textbox.setText(file_name)
        self.label2.setText(name[-1])
        self.label2.adjustSize()
        self.button2.show()


class SignMain(QMainWindow):
    def __init__(self, parent=None):
        super(SignMain, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 700
        self.height = 600
        self.dialogs = list()
        self.imageFile = ""
        self.setWindowTitle("Sign Recognition")
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Creare textBox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 30)

        # Creare buton de incarcare fisier
        self.button1 = self.createButton(300, 20, 100, 30, 'Incarcare fisier')
        # adaugarea actiunii butonului => functia uploadFile
        self.button1.clicked.connect(self.uploadFile)

        self.label2 = QLabel(self)
        self.label2.move(20, 90)
        self.label3 = QLabel(self)
        self.lb = QLabel(self)
        self.lbPredict = QLabel(self)
        self.predictBut = self.createButton(200, 120, 100, 30, "Predict")
        self.predictBut.hide()
        self.predictBut.clicked.connect(self.predict)

        self.button5 = self.createButton(300, 500, 200, 50, 'Exit')
        self.button5.clicked.connect(self.doNothig)

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def doNothig(self):
        self.close()

    def predict(self):
        image = Image.open(self.imageFile)
        sign, pred = classify(image)
        image = cv2.imread('/home/adrian/Documents/licenta/HUD/src/signRecognition/signs/' + str(pred + 1) + '.png')
        image = cv2.resize(image, (160, 160))
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.lbPredict.resize(160, 160)
        self.lbPredict.move(320, 120)
        self.lbPredict.setPixmap(QPixmap.fromImage(image))
        self.label3.move(370, 100)
        self.label3.setText(sign)
        self.label3.adjustSize()

    def uploadFile(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        name = file_name.split('/')
        self.textbox.setText(file_name)
        self.label2.setText(name[-1])
        self.label2.adjustSize()
        self.imageFile = file_name
        pixmap = QPixmap(file_name)
        self.lb.resize(160, 160)
        self.lb.move(20, 120)
        self.lb.setPixmap(pixmap.scaled(self.lb.size(), Qt.IgnoreAspectRatio))
        self.predictBut.show()


class UpdateDiagContinue(QMainWindow):
    def __init__(self, parent=None):
        super(UpdateDiagContinue, self).__init__(parent)
        self.left = 500
        self.top = 50
        self.width = 600
        self.height = 200
        self.dialogs = list()
        self.setWindowTitle("Update")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.text = self.createLabel(60, 20, 'Doriti sa efectuam verificarea versiunii?\n Poate dura cateva momente')
        self.box = QCheckBox("Doriti sa salvati preferintele dumneavoastra?", self)
        self.box.move(200, 20)
        self.box.resize(340, 300)
        self.button1 = self.createButton(130, 60, 100, 50, 'Da')
        self.button1.clicked.connect(self.confirmUpdate)
        self.button2 = self.createButton(250, 60, 100, 50, 'Nu')
        self.button2.clicked.connect(self.doNothingAndSave)
        self.text2 = self.createLabel(60, 20, "Va rog asteptati cateva momente")
        self.text2.hide()
        self.text3 = self.createLabel(60, 70, "O noua versiune este gata sa fie descarcata")
        self.text3.hide()
        self.text4 = self.createLabel(60, 70, "Descarcare reusita")
        self.text4.hide()
        self.button3 = self.createButton(130, 130, 100, 50, "Yes")
        self.button3.clicked.connect(self.confirmDownload)
        self.button3.hide()
        self.button4 = self.createButton(250, 130, 100, 50, "No")
        self.button4.clicked.connect(self.doNothig)
        self.button4.hide()
        self.button5 = self.createButton(200, 130, 100, 50, 'Ok')
        self.button5.clicked.connect(self.continueNormalExecution)
        self.button5.hide()

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def doNothig(self):
        self.close()

    def doNothingAndSave(self):
        data = []
        if self.box.isChecked():
            with open("preference.json", 'w') as output:
                data.append({'checkUpdate': 'no'})
                json.dump(data, output)
        self.close()

    def confirmUpdate(self):
        self.button1.hide()
        self.button2.hide()
        self.text.hide()
        self.box.hide()
        self.text3.show()
        self.button3.show()
        self.button4.show()

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def confirmDownload(self):
        self.button3.hide()
        self.button4.hide()
        self.text2.hide()
        self.text3.hide()
        self.text4.show()
        self.button5.show()

    def continueNormalExecution(self):
        dialog = SignMain(self)
        self.dialogs.append(dialog)
        dialog.show()
        self.close()


class UpdateDiag(QMainWindow):
    def __init__(self, parent=None):
        super(UpdateDiag, self).__init__(parent)
        self.left = 500
        self.top = 50
        self.width = 800
        self.height = 200
        self.setWindowTitle("Update")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.box = QCheckBox("Doriti sa verificati actualizarile la initierea aplicatiei?", self)
        self.box.move(200, 20)
        self.box.resize(240, 300)
        self.text = self.createLabel(60, 20, 'Do you want to check for an update?\n It might take a few minutes')
        self.button1 = self.createButton(130, 60, 100, 50, 'Yes')
        self.button1.clicked.connect(self.confirmUpdate)
        self.button2 = self.createButton(250, 60, 100, 50, 'No')
        self.button2.clicked.connect(self.doNothig)
        self.text2 = self.createLabel(60, 20, "Please wait while we check")
        self.text2.hide()
        self.text3 = self.createLabel(60, 70, "An update is ready to be downloaded")
        self.text3.hide()
        self.text4 = self.createLabel(60, 70, "Download successful")
        self.text4.hide()
        self.button3 = self.createButton(130, 130, 100, 50, "Yes")
        self.button3.clicked.connect(self.confirmDownload)
        self.button3.hide()
        self.button4 = self.createButton(250, 130, 100, 50, "No")
        self.button4.clicked.connect(self.doNothig)
        self.button4.hide()
        self.button5 = self.createButton(200, 130, 100, 50, 'Ok')
        self.button5.clicked.connect(self.doNothig)
        self.button5.hide()

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def doNothig(self):
        self.close()

    def confirmUpdate(self):
        data = []
        if self.box.isChecked():
            with open("preference.json", 'w') as output:
                data.append({'checkUpdate': 'yes'})
                json.dump(data, output)
        self.button1.hide()
        self.button2.hide()
        self.text.hide()
        self.text3.show()
        self.button3.show()
        self.button4.show()

    def createLabel(self, y_move, x_move, text):
        label = QLabel(self)
        label.move(y_move, x_move)
        label.setText(text)
        label.adjustSize()
        return label

    def confirmDownload(self):
        self.button3.hide()
        self.button4.hide()
        self.text2.hide()
        self.text3.hide()
        self.text4.show()
        self.button5.show()


class CloseDiag(QMainWindow):
    def __init__(self, main, parent=None):
        super(CloseDiag, self).__init__(parent)
        self.left = 500
        self.top = 50
        self.width = 500
        self.height = 200
        self.main = main
        self.setWindowTitle("Are you sure you want to close the app?")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.button1 = self.createButton(130, 60, 100, 50, 'Yes')
        self.button1.clicked.connect(self.confirmExit)
        self.button2 = self.createButton(250, 60, 100, 50, 'No')
        self.button2.clicked.connect(self.doNothig)

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def confirmExit(self):
        self.iesire = 1
        self.close()
        self.main.close()

    def doNothig(self):
        self.close()


class App(QMainWindow):

    def __init__(self):
        # Creare fereastra de aplicatie
        super().__init__()
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
        self.dialogs = list()

        self.setWindowTitle("Head-Up Display")
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Creare textBox
        # self.textbox = QLineEdit(self)
        # self.textbox.move(20, 20)
        # self.textbox.resize(280, 30)

        self.button1 = self.createButton(30, 20, 500, 50, 'Recunoasterea semnelor de circulatie')
        # adaugarea actiunii butonului => functia uploadFile
        self.button1.clicked.connect(self.signRecog)

        self.button2 = self.createButton(30, 80, 500, 50, 'Recunoasterea benzii de mers')
        self.button2.clicked.connect(self.laneDialog)

        self.button3 = self.createButton(30, 300, 200, 50, 'Mod Admin')
        self.button3.clicked.connect(self.doNothig)

        self.button4 = self.createButton(30, 200, 500, 50, 'Update')
        self.button4.clicked.connect(self.updateDialog)

        self.button5 = self.createButton(300, 300, 200, 50, 'Exit')
        self.button5.clicked.connect(self.closeDialog)

        self.button6 = self.createButton(30, 140, 500, 50, 'Cautare semn de circulatie')
        self.button6.clicked.connect(self.searchSign)

        self.button7 = self.createButton(30, 250, 500, 50, 'Accident')
        self.button7.clicked.connect(self.mainDiag)

        self.textbox = QLineEdit(self)
        self.textbox.hide()

    def uploadFile(self):
        self.textbox.show()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.textbox.setText(file_name)

    def doNothig(self):
        print("nothing")

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def closeDialog(self):
        dialog = CloseDiag(self, self)
        self.dialogs.append(dialog)
        dialog.show()

    def updateDialog(self):
        dialog = UpdateDiag(self)
        self.dialogs.append(dialog)
        dialog.show()

    def laneDialog(self):
        dialog = LaneMain(self)
        self.dialogs.append(dialog)
        dialog.show()

    def mainDiag(self):
        dialog = MainDialog(self)
        self.dialogs.append(dialog)
        dialog.show()

    def searchSign(self):
        dialog = SearchSign(self)
        self.dialogs.append(dialog)
        dialog.show()

    def signRecog(self):
        with open('preference.json') as json_file:
            data = json.load(json_file)
            print(data[0]['checkUpdate'])
        if data[0]['checkUpdate'] == 'no':
            print("sal")
            dialog = SignMain(self)
        else:
            dialog = UpdateDiagContinue(self)
        self.dialogs.append(dialog)
        dialog.show()


if __name__ == '__main__':
    # laneR("../laneRecognition/test_video/test2.mp4")
    load_model()
    app = QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    ex = App()
    ex.show()
    app.exec_()
