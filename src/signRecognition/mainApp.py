import sys
import json
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QFileDialog, QLabel, QDialog, \
    QGridLayout, QCheckBox


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
        # adaugarea actiunii butonului => functia uploadFile
        self.button1.clicked.connect(self.uploadFile)
        self.label2 = QLabel(self)
        self.label2.move(20, 90)

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
        name = file_name.split('/')
        self.textbox.setText(file_name)
        self.label2.setText(name[-1])
        self.label2.adjustSize()


class SignMain(QMainWindow):
    def __init__(self, parent=None):
        super(SignMain, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
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

    def closeDialog(self):
        dialog = CloseDiag(self, self)
        self.dialogs.append(dialog)
        dialog.show()

    def predict(self):
        pixmap2 = QPixmap(self.imageFile)
        self.lbPredict.resize(160, 160)
        self.lbPredict.move(320, 120)
        self.lbPredict.setPixmap(pixmap2.scaled(self.lbPredict.size(), Qt.IgnoreAspectRatio))
        self.label3.move(370, 100)
        self.label3.setText(self.imageFile)
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
        self.width = 500
        self.height = 200
        self.dialogs = list()
        self.setWindowTitle("Update")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.text = self.createLabel(60, 20, 'Do you want to check for an update?\n It might take a few minutes')
        self.box = QCheckBox("Do you want to save this option?", self)
        self.box.move(200, 20)
        self.box.resize(240, 300)
        self.button1 = self.createButton(130, 60, 100, 50, 'Yes')
        self.button1.clicked.connect(self.confirmUpdate)
        self.button2 = self.createButton(250, 60, 100, 50, 'No')
        self.button2.clicked.connect(self.doNothingAndSave)
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
        data=[]
        if self.box.isChecked():
            with open("preference.json",'w') as output:
                data.append({'checkUpdate':'no'})
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
        self.width = 500
        self.height = 200
        self.setWindowTitle("Update")
        self.setGeometry(self.left, self.top, self.width, self.height)
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

        self.button4 = self.createButton(30, 140, 500, 50, 'Update')
        self.button4.clicked.connect(self.updateDialog)

        self.button5 = self.createButton(300, 300, 200, 50, 'Exit')
        self.button5.clicked.connect(self.closeDialog)
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

    def signRecog(self):
        with open('preference.json') as json_file:
            data = json.load(json_file)
            print(data[0]['checkUpdate'])
        if data[0]['checkUpdate'] == 'no':
            print("sal")
            dialog=SignMain(self)
        else:
            dialog = UpdateDiagContinue(self)
        self.dialogs.append(dialog)
        dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    ex = App()
    ex.show()
    app.exec_()
