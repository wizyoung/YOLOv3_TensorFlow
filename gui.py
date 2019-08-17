import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(300, 200))
        self.setWindowTitle("PyCam")

        pybutton_1 = QPushButton('Start Detector', self)
        pybutton_1.clicked.connect(self.start_detector)
        pybutton_1.resize(100,32)
        pybutton_1.move(50, 50)

        pybutton_2 = QPushButton('View Images', self)
        pybutton_2.clicked.connect(self.start_detector)
        pybutton_2.resize(100,32)
        pybutton_2.move(50, 100)

    def clickMethod(self):
        print('kys nigger')
    def start_detector(self):
        os.system('python3 camera_test.py --camera 2 --new_size 150 100')

    def view_images(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )
