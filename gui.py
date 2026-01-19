from train import train
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import *


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Number Neural Network")

        button = QPushButton("Draw!")

        self.setMinimumSize(QSize(400, 300))

        self.setCentralWidget(button)



app = QApplication(sys.argv)
# sys.argv is python list containing command line arguments. if no command line needed, paste empty list []
# QApplication holds event loop

window = MainWindow()
window.show()

app.exec()
