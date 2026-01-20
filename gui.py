import sys

from PyQt6.QtGui import *
from PyQt6.QtCore import *

from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
    QCheckBox,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QLabel, # One line text cell
    QLineEdit,
    QListWidget,
    QSlider,
    QSpinBox,
)



from train import training


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Number Neural Network")
        self.setGeometry(400, 200, 500, 700)
        self.setMinimumSize(QSize(300, 500))

        #draw_button = QPushButton("Draw!")
        #draw_button.clicked.connect(self.draw_button_clicked)
        #self.setCentralWidget(draw_button)

        image_label = QLabel()

        image = QPixmap("network_image.jpg")
        image_label.setPixmap(image)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(image_label)


app = QApplication(sys.argv)
# sys.argv is python list containing command line arguments. if no command line needed, paste empty list []
# QApplication holds event loop

window = MainWindow()
window.show()

app.exec()
