import cmd
import os
import sys
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QFileDialog, QLabel, QDialog, QGridLayout
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


class UpdateDiag(QMainWindow):
    def __init__(self, parent=None):
        super(UpdateDiag, self).__init__(parent)
        self.left = 500
        self.top = 50
        self.width = 500
        self.height = 200
        self.iesire = 0
        self.setWindowTitle("Update")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.text = QLabel(self)
        self.text.move(60, 20)
        self.text.setText('Do you want to check for an update?\n It might take a few minutes')
        self.text.adjustSize()
        self.button1 = self.createButton(130, 60, 100, 50, 'Yes')
        self.button1.clicked.connect(self.confirmUpdate)
        self.button2 = self.createButton(250, 60, 100, 50, 'No')
        self.button2.clicked.connect(self.doNothig)
        self.text2 = QLabel(self)
        self.text3 = QLabel(self)
        self.text4 = QLabel(self)
        self.button3 = QPushButton("Yes sir",self)
        self.button3.clicked.connect(self.confirmDownload)
        self.button4 = QPushButton("No sir", self)
        self.button4.clicked.connect(self.doNothig)
        self.button3.hide()
        self.button4.hide()

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def doNothig(self):
        self.close()

    def confirmUpdate(self):
        self.button1.hide()
        self.button2.hide()
        self.text.hide()
        self.text2.move(60, 20)
        self.text2.setText('Please wait while we check')
        self.text2.adjustSize()
        # self.text2.hide()
        self.text3.move(60, 70)
        self.text3.setText('An update is ready to be downloaded')
        self.text3.adjustSize()
        self.button3.move(130,130)
        self.button3.resize(100,50)
        self.button3.show()
        self.button4.move(250, 130)
        self.button4.resize(100, 50)
        self.button4.show()

    def confirmDownload(self):
        self.button3.hide()
        self.button4.hide()
        self.text2.hide()
        self.text3.hide()
        self.text4.move(60, 70)
        self.text4.setText('Download successful')
        self.text4.adjustSize()
        self.button4.show()


class CloseDiag(QMainWindow):
    def __init__(self, main, parent=None):
        super(CloseDiag, self).__init__(parent)
        self.left = 500
        self.top = 50
        self.width = 500
        self.height = 200
        self.iesire = 0
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
        self.button1.clicked.connect(self.uploadFile)

        self.button2 = self.createButton(30, 80, 500, 50, 'Recunoasterea benzii de mers')
        self.button2.clicked.connect(self.doNothig)

        self.button3 = self.createButton(30, 300, 200, 50, 'Mod Admin')
        self.button3.clicked.connect(self.doNothig)

        self.button4 = self.createButton(30, 140, 500, 50, 'Update')
        self.button4.clicked.connect(self.updateDialog)

        self.button5 = self.createButton(300, 300, 200, 50, 'Exit')
        self.button5.clicked.connect(self.closeDialog)

    def uploadFile(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', 'C:\\', "Audio files (*.jpg)")
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    ex = App()
    ex.show()
    app.exec_()
