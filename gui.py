import sys

from PyQt6.QtGui import *
from PyQt6.QtCore import *

from PyQt6.QtWidgets import *


class Pixel(QWidget):
    
    def __init__(self):

        super().__init__()
        self.color = QColor("white")
        #self.setFixedSize(15, 15)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.color)

    def paint_black(self):
        if self.color != QColor("black"):
            self.color = QColor("black")
            self.update()

    def clear(self):
        self.color = QColor("white")
        self.update()



class PixelCanvas(QWidget):

    def __init__(self, pixels, grid):
        super().__init__()
        self.pixels = pixels
        self.grid = grid
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.paint_at(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.paint_at(event.position().toPoint())

    def paint_at(self, pos):
        widget = self.childAt(pos)
        if isinstance(widget, Pixel):
            widget.paint_black()


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
        TrainButton.pressed.connect(self.TrainButton_Pressed)

        MainMenuWidget = QWidget()
        MainMenuWidget.setLayout(MainMenuLayout)


        # Canvas
        Canvas = QGridLayout()
        Canvas.setSpacing(0)
        Canvas.setContentsMargins(0, 0, 0, 0)
        
        self.pixels = [Pixel() for _ in range(PIXELS**2)]

        idx = 0
        for row in range(PIXELS):
            for col in range(PIXELS):
                Canvas.addWidget(self.pixels[idx], row, col)
                idx += 1



        # Draw Page
        DrawLayout = QVBoxLayout()

        DrawField = PixelCanvas(self.pixels, Canvas)
        DrawField.setLayout(Canvas)
        DrawLayout.addWidget(DrawField)

        ClearButton = QPushButton("Clear")
        DrawLayout.addWidget(ClearButton)
        ClearButton.pressed.connect(self.ClearButton_Pressed)

        GuessButton = QPushButton("Guess the number")
        DrawLayout.addWidget(GuessButton)
        GuessButton.pressed.connect(self.GuessButton_Pressed)

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

    def TrainButton_Pressed(self):
        pass


    def ClearButton_Pressed(self):
        for pixel in self.pixels:
            pixel.color = QColor("white")
            pixel.update()

    def GuessButton_Pressed(self):
        pass
    
    def BackButton_Pressed(self):
        self.Layout.setCurrentIndex(0)

    

PIXELS = 28


app = QApplication(sys.argv)
# sys.argv is python list containing command line arguments. if no command line needed, paste empty list []
# QApplication holds event loop

window = MainWindow()
window.show()

app.exec()
