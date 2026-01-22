import sys

from PyQt6.QtGui import *
from PyQt6.QtCore import *

from PyQt6.QtWidgets import *


class Pixel(QWidget):
    
    def __init__(self, color, row, col):
        super().__init__()

        self.row = row
        self.col = col

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

    def mousePressEvent(self, e):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("black"))
        self.setPalette(palette)
        print(self.row, self.col)



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Number Neural Network")
        self.setGeometry(400, 200, 500, 500)
        self.setFixedSize(QSize(500, 500))

        # Main Menu
        MainMenuLayout = QVBoxLayout()
        
        InfoTextLabel = QLabel()
        InfoTextLabel.setText("Info text.")
        MainMenuLayout.addWidget(InfoTextLabel)

        DrawButton = QPushButton("Draw!")
        MainMenuLayout.addWidget(DrawButton)
        DrawButton.pressed.connect(self.DrawButton_Pressed)

        TrainButton = QPushButton("Training")
        MainMenuLayout.addWidget(TrainButton)

        MainMenuWidget = QWidget()
        MainMenuWidget.setLayout(MainMenuLayout)


        # Draw Field
        Canvas = QGridLayout()
        for row in range(PIXELS):
            for col in range(PIXELS):
                Canvas.addWidget(Pixel("white", row, col), row, col)
        Canvas.setSpacing(False)



        # Draw Page
        DrawLayout = QVBoxLayout()

        DrawField = QWidget()
        DrawField.setLayout(Canvas)
        DrawLayout.addWidget(DrawField)

        ClearButton = QPushButton("Clear")
        DrawLayout.addWidget(ClearButton)

        GuessButton = QPushButton("Guess the number")
        DrawLayout.addWidget(GuessButton)

        BackButton = QPushButton("Back")
        DrawLayout.addWidget(BackButton)
        BackButton.pressed.connect(self.BackButton_Pressed)

        DrawWidget = QWidget()
        DrawWidget.setLayout(DrawLayout)


        # Overall layout
        self.Layout = QStackedLayout()
        self.Layout.addWidget(MainMenuWidget)
        self.Layout.addWidget(DrawWidget)

        # Overall widget
        Widget = QWidget()
        Widget.setLayout(self.Layout)
        self.setCentralWidget(Widget)



    def DrawButton_Pressed(self):
        self.Layout.setCurrentIndex(1)

    def BackButton_Pressed(self):
        self.Layout.setCurrentIndex(0)


PIXELS = 28


app = QApplication(sys.argv)
# sys.argv is python list containing command line arguments. if no command line needed, paste empty list []
# QApplication holds event loop

window = MainWindow()
window.show()

app.exec()
