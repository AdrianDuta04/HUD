
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QFileDialog, QLabel, QDialog, QGridLayout


class SignMain(QMainWindow):
    def __init__(self, parent=None):
        super(SignMain, self).__init__(parent)
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
        self.dialogs = list()
        self.imageFile=""
        self.setWindowTitle("Sign Recognition")
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Creare textBox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 30)

        # Creare buton de incarcare fisier
        self.button1 = QPushButton('Incarcare fisier', self)
        self.button1.move(300, 20)
        self.button1.resize(100, 30)
        # adaugarea actiunii butonului => functia uploadFile
        self.button1.clicked.connect(self.uploadFile)

        self.label2 = QLabel(self)
        self.label2.move(20, 90)

        # self.label3 = QLabel(self)
        # pixmap = QPixmap('image.jpeg')
        # self.label3.setPixmap(pixmap)
        # self.label3.move(100,100)

    def createButton(self, y_move, x_move, y_size, x_size, text):
        button = QPushButton(text, self)
        button.move(y_move, x_move)
        button.resize(y_size, x_size)
        return button

    def createLabel(self,y_move,x_move,text):
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

    def uploadFile(self):
        self.textbox = QLineEdit(self)
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        name=file_name.split('/')
        self.textbox.setText(file_name)
        self.label2.setText(name[-1])
        self.label2.adjustSize()


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
        self.button1 = self.createButton(130, 60, 100, 50, 'Yes')
        self.button1.clicked.connect(self.confirmUpdate)
        self.button2 = self.createButton(250, 60, 100, 50, 'No')
        self.button2.clicked.connect(self.doNothig)
        self.text2 = self.createLabel(60,20,"Please wait while we check")
        self.text2.hide()
        self.text3 = self.createLabel(60,70,"An update is ready to be downloaded")
        self.text3.hide()
        self.text4 = self.createLabel(60, 70, "Download successful")
        self.text4.hide()
        self.button3=self.createButton(130,130,100,50,"Yes")
        self.button3.clicked.connect(self.confirmDownload)
        self.button3.hide()
        self.button4 = self.createButton(250, 130, 100, 50, "No")
        self.button4.clicked.connect(self.doNothig)
        self.button4.hide()
        self.button5=self.createButton(200,130,100,50,'Ok')
        self.button5.clicked.connect(self.continueNormalExecution)
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

    def createLabel(self,y_move,x_move,text):
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
        self.text2 = self.createLabel(60,20,"Please wait while we check")
        self.text2.hide()
        self.text3 = self.createLabel(60,70,"An update is ready to be downloaded")
        self.text3.hide()
        self.text4 = self.createLabel(60, 70, "Download successful")
        self.text4.hide()
        self.button3=self.createButton(130,130,100,50,"Yes")
        self.button3.clicked.connect(self.confirmDownload)
        self.button3.hide()
        self.button4 = self.createButton(250, 130, 100, 50, "No")
        self.button4.clicked.connect(self.doNothig)
        self.button4.hide()
        self.button5=self.createButton(200,130,100,50,'Ok')
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

    def createLabel(self,y_move,x_move,text):
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

    def signRecog(self):
        dialog = UpdateDiagContinue(self)
        self.dialogs.append(dialog)
        dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    ex = App()
    ex.show()
    app.exec_()
