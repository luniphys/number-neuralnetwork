import sys
import numpy as np
import pandas as pd

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtCharts import *

from train import getActivations



def getGuess(sample, alrTrained=True):

    if alrTrained:
        w1 = np.array(pd.read_csv("TrainedWBs/Trained_train/w1.csv"))
        b1 = np.array(pd.read_csv("TrainedWBs/Trained_train/b1.csv"))
        w2 = np.array(pd.read_csv("TrainedWBs/Trained_train/w2.csv"))
        b2 = np.array(pd.read_csv("TrainedWBs/Trained_train/b2.csv"))
        w3 = np.array(pd.read_csv("TrainedWBs/Trained_train/w3.csv"))
        b3 = np.array(pd.read_csv("TrainedWBs/Trained_train/b3.csv"))

    else:
        w1 = np.array(pd.read_csv("WeightsBiases/w1.csv"))
        b1 = np.array(pd.read_csv("WeightsBiases/b1.csv"))
        w2 = np.array(pd.read_csv("WeightsBiases/w2.csv"))
        b2 = np.array(pd.read_csv("WeightsBiases/b2.csv"))
        w3 = np.array(pd.read_csv("WeightsBiases/w3.csv"))
        b3 = np.array(pd.read_csv("WeightsBiases/b3.csv"))

    if sum(sample) == 0:
        return None, [0] * 10

    a3 = getActivations(sample, w1, b1, w2, b2, w3, b3, False)[4]
    perc_lst = [round(float(num * 100), 2) for num in a3]

    return max(enumerate(a3), key = lambda x: x[1])[0], perc_lst



class Canvas(QWidget):

    def __init__(self):
        super().__init__()
        self.window_size = 500
        self.setFixedSize(QSize(self.window_size, self.window_size))
        self.PIXELSIZE = 28
        self.pixels = [0 for _ in range(self.PIXELSIZE**2)]
        self.length = round(self.window_size / self.PIXELSIZE)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        idx = 0
        for row in range(self.PIXELSIZE):
            for col in range(self.PIXELSIZE):
                if self.pixels[idx] == 0:
                    painter.fillRect(col * self.length, row * self.length, self.length + 1, self.length + 1, QColor("white"))
                if self.pixels[idx] == 1:
                    painter.fillRect(col * self.length, row * self.length, self.length + 1, self.length + 1, QColor("black"))
                idx += 1

    def mouseMoveEvent(self, event):
        self.paint(event)

    def mousePressEvent(self, event):
        self.paint(event)

    def paint(self, event):
        x = int(event.position().x() // self.length)
        y = int(event.position().y() // self.length)
        if 0 <= x < self.PIXELSIZE and 0 <= y < self.PIXELSIZE:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.pixels[y * self.PIXELSIZE + x] = 1
                self.pixels[y * self.PIXELSIZE + x + 1] = 1
                self.pixels[y * self.PIXELSIZE + x - 1] = 1
                self.pixels[y * self.PIXELSIZE + x + self.PIXELSIZE] = 1
                self.pixels[y * self.PIXELSIZE + x - self.PIXELSIZE] = 1
            elif event.buttons() & Qt.MouseButton.RightButton:
                self.pixels[y * self.PIXELSIZE + x] = 0
            self.update()

    def clearAll(self):
        for idx in range(self.PIXELSIZE**2):
            self.pixels[idx] = 0
        self.update()

    def getPixels(self):
        return self.pixels



class ProbabilityBarChart(QWidget):

    def __init__(self):
        super().__init__()

        self.barset = QBarSet("Probabilities")
        self.barset.append([0] * 10)

        self.series = QBarSeries()
        self.series.append(self.barset)

        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        self.chart.legend().hide()

        categories = [f"{i}" for i in range(10)]
        x_axis = QBarCategoryAxis()
        x_axis.append(categories)
        self.chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(x_axis)

        y_axis = QValueAxis()
        y_axis.setRange(0, 100)
        y_axis.setTickCount(6)
        self.chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(y_axis)

        chartview = QChartView(self.chart)
        chartview.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = QVBoxLayout()
        layout.addWidget(chartview)
        self.setLayout(layout)
    
    def updateValues(self, values):
        for idx, val in enumerate(values):
            self.barset.replace(idx, val)





class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Number Neural Network")
        self.setGeometry(900, 80, 520, 850) # position, size
        self.setFixedSize(QSize(520, 850))

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

        ExitButtonMenu = QPushButton("Exit")
        MainMenuLayout.addWidget(ExitButtonMenu)
        ExitButtonMenu.pressed.connect(self.close)

        MainMenuWidget = QWidget()
        MainMenuWidget.setLayout(MainMenuLayout)

        # Draw Page
        DrawPageLayout = QVBoxLayout()

        self.DrawField = Canvas()
        DrawPageLayout.addWidget(self.DrawField)

        GuessButton = QPushButton("Guess the number!")
        DrawPageLayout.addWidget(GuessButton)
        GuessButton.pressed.connect(self.GuessButton_Pressed)

        ClearButton = QPushButton("Clear")
        DrawPageLayout.addWidget(ClearButton)
        ClearButton.pressed.connect(self.ClearButton_Pressed)

        self.GuessAnswer = QLabel()
        DrawPageLayout.addWidget(self.GuessAnswer)
        self.GuessAnswer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.BarChart = ProbabilityBarChart()
        DrawPageLayout.addWidget(self.BarChart)

        BackExitBottomWidget = QWidget()
        BackExitBottomLayout = QHBoxLayout()

        BackButton = QPushButton("Back")
        BackExitBottomLayout.addWidget(BackButton)
        BackButton.pressed.connect(self.BackButton_Pressed)

        ExitButtonDraw = QPushButton("Exit")
        BackExitBottomLayout.addWidget(ExitButtonDraw)
        ExitButtonDraw.pressed.connect(self.close)

        BackExitBottomWidget.setLayout(BackExitBottomLayout)
        DrawPageLayout.addWidget(BackExitBottomWidget)

        DrawPageWidget = QWidget()
        DrawPageWidget.setLayout(DrawPageLayout)


        # Overall layout
        self.Layout = QStackedLayout()
        self.Layout.addWidget(MainMenuWidget)
        self.Layout.addWidget(DrawPageWidget)

        # Overall widget
        Widget = QWidget()
        Widget.setLayout(self.Layout)
        self.setCentralWidget(Widget)



    def DrawButton_Pressed(self):
        self.Layout.setCurrentIndex(1)

    def TrainButton_Pressed(self):
        pass


    def ClearButton_Pressed(self):
        self.DrawField.clearAll()
        self.GuessAnswer.setText("")
        self.BarChart.updateValues([0] * 10)

    def GuessButton_Pressed(self):
        pixels = self.DrawField.getPixels()

        guessed_num, perc_lst = getGuess(pixels)

        self.GuessAnswer.setText("Guess: " + str(guessed_num))
        self.BarChart.updateValues(perc_lst)

    
    def BackButton_Pressed(self):
        self.Layout.setCurrentIndex(0)

    def ExitButton_Pressed(self):
        pass





if __name__ == "__main__":

    PIXELSIZE = 28


    app = QApplication(sys.argv)
    # sys.argv is python list containing command line arguments. if no command line needed, paste empty list []
    # QApplication holds event loop

    window = MainWindow()
    window.show()

    app.exec()
