import os
import sys
import shutil
import numpy as np
import pandas as pd
import json
from threading import Thread

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtCharts import *

from PyQt6.QtSvgWidgets import QSvgWidget

from train import makeRandomWeightsBiases, getActivations, training



def getGuess(sample, alrTrained):

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
        self.PIXELSIZE = 28
        self.pixels = [0.0 for _ in range(self.PIXELSIZE**2)]
        self.length = round(self.window_size / self.PIXELSIZE)
        self.setMouseTracking(True)
        self.setMinimumSize(0, 0)
        self.lastPos = None
        self.BrushRadius = 1


    def paintEvent(self, event):
        painter = QPainter(self)
        self.length = round(min(self.width(), self.height()) / self.PIXELSIZE)

        idx = 0
        for row in range(self.PIXELSIZE):
            for col in range(self.PIXELSIZE):
                color = QColor("black") if self.pixels[idx] > 0 else QColor("white")
                painter.fillRect(col * self.length, row * self.length, self.length + 1, self.length + 1, color)
                idx += 1

                # Grid
                #painter.setPen(QPen(QColor("lightgray"), 1))
                #painter.drawRect(col * self.length, row * self.length, self.length, self.length)


    def mouseMoveEvent(self, event):
        if self.lastPos is not None:
            self.interpolate_paint(self.lastPos, event.position(), event)
        self.lastPos = event.position()


    def mousePressEvent(self, event):
        self.lastPos = event.position()
        self.paint(event)


    def paint(self, event):
        self.length = round(min(self.width(), self.height()) / self.PIXELSIZE)

        x = int(event.position().x() // self.length)
        y = int(event.position().y() // self.length)

        if 0 <= x < self.PIXELSIZE and 0 <= y < self.PIXELSIZE:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.drawBrush(x, y)
            elif event.buttons() & Qt.MouseButton.RightButton:
                self.pixels[y * self.PIXELSIZE + x] = 0
            self.update()


    def interpolate_paint(self, start_pos, end_pos, event):
        self.length = round(min(self.width(), self.height()) / self.PIXELSIZE)

        dx = end_pos.x() - start_pos.x()
        dy = end_pos.y() - start_pos.y()
        distance = (dx**2 + dy**2)**0.5

        if distance < 1:
            return
        
        steps = max(int(distance) + 1, 2)

        for i in range(steps):
            t = i / steps
            x = int((start_pos.x() + dx * t) // self.length)
            y = int((start_pos.y() + dy * t) // self.length)

            if 0 <= x < self.PIXELSIZE and 0 <= y < self.PIXELSIZE:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    self.drawBrush(x, y)
                elif event.buttons() & Qt.MouseButton.RightButton:
                    self.pixels[y * self.PIXELSIZE + x] = 0
                self.update()


    def drawBrush(self, x, y):
        for dx in range(-self.BrushRadius * 2, self.BrushRadius * 2 + 1):
            for dy in range(-self.BrushRadius * 2, self.BrushRadius * 2 + 1):
                distance = (dx**2 + dy**2)**0.5

                if distance <= self.BrushRadius:
                    intensity = np.exp(-(distance**2) / (2 * self.BrushRadius**2))
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.PIXELSIZE and 0 <= ny < self.PIXELSIZE:
                        idx = ny * self.PIXELSIZE + nx
                        self.pixels[idx] = max(self.pixels[idx], intensity)
    

    def clearAll(self):
        self.pixels = [0 for _ in range(self.PIXELSIZE**2)]
        self.update()


    def applySmoothing(self, pixels_array):
        smoothed = np.copy(pixels_array)
        
        for i in range(self.PIXELSIZE):
            for j in range(self.PIXELSIZE):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.PIXELSIZE and 0 <= nj < self.PIXELSIZE:
                            neighbors.append(pixels_array[ni, nj])
                smoothed[i, j] = np.mean(neighbors)
        
        return smoothed


    def getPixels(self):
        pixels_array = np.array(self.pixels, dtype=np.float32).reshape((self.PIXELSIZE, self.PIXELSIZE))

        pixels_array = self.applySmoothing(pixels_array)

        rows = np.any(pixels_array, axis=1)
        cols = np.any(pixels_array, axis=0)
        
        if not rows.any() or not cols.any():
            return np.zeros(784).tolist()
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        digit = pixels_array[rmin:rmax+1, cmin:cmax+1]
        digit_h, digit_w = digit.shape
        
        max_dim = max(digit_h, digit_w)
        pad_h = (max_dim - digit_h) // 2
        pad_w = (max_dim - digit_w) // 2
        
        digit_square = np.zeros((max_dim, max_dim), dtype=np.float32)
        digit_square[pad_h:pad_h+digit_h, pad_w:pad_w+digit_w] = digit
        
        target_size = 20
        if max_dim > target_size:
            new_size = target_size
            binned = np.zeros((new_size, new_size), dtype=np.float32)
            bin_size = max_dim / new_size
            for i in range(new_size):
                for j in range(new_size):
                    start_i = int(i * bin_size)
                    end_i = int((i + 1) * bin_size)
                    start_j = int(j * bin_size)
                    end_j = int((j + 1) * bin_size)
                    binned[i, j] = digit_square[start_i:end_i, start_j:end_j].mean()
            digit_square = binned
        
        final = np.zeros((self.PIXELSIZE, self.PIXELSIZE), dtype=np.float32)
        offset = (self.PIXELSIZE - digit_square.shape[0]) // 2
        final[offset:offset+digit_square.shape[0], offset:offset+digit_square.shape[1]] = digit_square
        
        if final.max() > 0:
            final = final / final.max()
        
        return final.flatten().tolist()



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



class CostPlot(QWidget):
    pass



# Qt Designer Code
class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        if os.path.exists("WeightsBiases/cycles.json"):
            with open("WeightsBiases/cycles.json", "r", encoding="utf-8") as file:
                self.CycleNum = json.load(file).get("cycles", 0)
        else:
            self.CycleNum = 0

        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(900, 50, 450, 870)
        font = QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        MainWindow.setMouseTracking(True)
        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QStackedWidget(parent=self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")

        selfTrainedDataExists = os.path.exists("WeightsBiases") \
                                and os.path.isfile("WeightsBiases/w1.csv") and os.path.isfile("WeightsBiases/b1.csv") \
                                and os.path.isfile("WeightsBiases/w1.csv") and os.path.isfile("WeightsBiases/b2.csv") \
                                and os.path.isfile("WeightsBiases/w3.csv") and os.path.isfile("WeightsBiases/b3.csv")

        # Main Menu
        self.MainMenuW = QWidget()
        self.MainMenuW.setObjectName("MainMenuW")
        self.verticalLayout_2 = QVBoxLayout(self.MainMenuW)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.MainMenuL = QVBoxLayout()
        self.MainMenuL.setSpacing(8)
        self.MainMenuL.setObjectName("MainMenuL")

        self.InfoLabel = QLabel(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.InfoLabel.sizePolicy().hasHeightForWidth())
        self.InfoLabel.setSizePolicy(sizePolicy)
        self.InfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.InfoLabel.setObjectName("InfoLabel")
        self.MainMenuL.addWidget(self.InfoLabel)

        self.DrawLayout = QHBoxLayout()
        self.DrawLayout.setObjectName("DrawLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawLayout.addItem(spacerItem)
        self.DrawButton = QPushButton(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DrawButton.sizePolicy().hasHeightForWidth())
        self.DrawButton.setSizePolicy(sizePolicy)
        self.DrawButton.setObjectName("DrawButton")
        self.DrawLayout.addWidget(self.DrawButton)
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawLayout.addItem(spacerItem1)
        self.MainMenuL.addLayout(self.DrawLayout)

        self.TrainingLayout = QHBoxLayout()
        self.TrainingLayout.setObjectName("TrainingLayout")
        spacerItem2 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingLayout.addItem(spacerItem2)
        self.TrainingButton = QPushButton(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainingButton.sizePolicy().hasHeightForWidth())
        self.TrainingButton.setSizePolicy(sizePolicy)
        self.TrainingButton.setObjectName("TrainingButton")
        self.TrainingLayout.addWidget(self.TrainingButton)
        spacerItem3 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingLayout.addItem(spacerItem3)
        self.MainMenuL.addLayout(self.TrainingLayout)

        spacerItem4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.MainMenuL.addItem(spacerItem4)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem5 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.ExitButtonMain = QPushButton(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ExitButtonMain.sizePolicy().hasHeightForWidth())
        self.ExitButtonMain.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(12)
        self.ExitButtonMain.setFont(font)
        self.ExitButtonMain.setObjectName("ExitButtonMain")
        self.horizontalLayout_2.addWidget(self.ExitButtonMain)
        spacerItem6 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.MainMenuL.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.MainMenuL)
        self.stackedWidget.addWidget(self.MainMenuW)

        # Draw Page
        self.DrawPageW = QWidget()
        self.DrawPageW.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.DrawPageW.setAutoFillBackground(False)
        self.DrawPageW.setObjectName("DrawPageW")
        self.verticalLayout_3 = QVBoxLayout(self.DrawPageW)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.DrawPageL = QVBoxLayout()
        self.DrawPageL.setSpacing(8)
        self.DrawPageL.setObjectName("DrawPageL")

        self.Canvas = Canvas()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.Canvas.sizePolicy().hasHeightForWidth())
        self.Canvas.setSizePolicy(sizePolicy)
        self.Canvas.setMinimumSize(QSize(200, 200))
        self.Canvas.setMaximumWidth(550)
        self.Canvas.setBaseSize(QSize(0, 0))
        self.Canvas.setMouseTracking(False)
        self.Canvas.setObjectName("Canvas")
        self.CanvasLayout = QHBoxLayout()
        self.CanvasLayout.setContentsMargins(0, 0, 0, 0)
        self.CanvasLayout.setSpacing(0)
        self.CanvasLayout.addStretch()
        self.CanvasLayout.addWidget(self.Canvas, 1)
        self.CanvasLayout.addStretch()
        self.DrawPageL.addLayout(self.CanvasLayout, 3)

        self.CanvasInfoLabel = QLabel(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(10)
        self.CanvasInfoLabel.setFont(font)
        self.CanvasInfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.CanvasInfoLabel.setObjectName("CanvasInfoLabel")
        self.DrawPageL.addWidget(self.CanvasInfoLabel)

        spacerItem69 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawPageL.addItem(spacerItem69)

        self.ClearGuessLayout = QHBoxLayout()
        self.ClearGuessLayout.setObjectName("ClearGuessLayout")
        spacerItem7 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem7)
        self.ClearButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ClearButton.sizePolicy().hasHeightForWidth())
        self.ClearButton.setSizePolicy(sizePolicy)
        self.ClearButton.setObjectName("ClearButton")
        self.ClearGuessLayout.addWidget(self.ClearButton)
        spacerItem8 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem8)
        self.GuessButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GuessButton.sizePolicy().hasHeightForWidth())
        self.GuessButton.setSizePolicy(sizePolicy)
        self.GuessButton.setObjectName("GuessButton")
        self.ClearGuessLayout.addWidget(self.GuessButton)
        spacerItem9 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem9)
        self.DrawPageL.addLayout(self.ClearGuessLayout)

        self.ResultLayout = QHBoxLayout()
        self.ResultLayout.setObjectName("ResultLayout")
        spacerItem10 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ResultLayout.addItem(spacerItem10)
        self.ResultLabel = QLabel(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ResultLabel.sizePolicy().hasHeightForWidth())
        self.ResultLabel.setSizePolicy(sizePolicy)
        self.ResultLabel.setFrameShape(QFrame.Shape.StyledPanel)
        self.ResultLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ResultLabel.setObjectName("ResultLabel")
        self.ResultLayout.addWidget(self.ResultLabel)
        spacerItem11 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ResultLayout.addItem(spacerItem11)
        self.DrawPageL.addLayout(self.ResultLayout)

        self.BarChart = ProbabilityBarChart()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.BarChart.sizePolicy().hasHeightForWidth())
        self.BarChart.setSizePolicy(sizePolicy)
        self.BarChart.setSizeIncrement(QSize(0, 0))
        self.BarChart.setObjectName("BarChart")
        self.DrawPageL.addWidget(self.BarChart)

        self.DataLayout = QHBoxLayout()
        self.DataLayout.setObjectName("DataLayout")
        spacerItem12 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DataLayout.addItem(spacerItem12)
        self.PretrainedButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PretrainedButton.sizePolicy().hasHeightForWidth())
        self.PretrainedButton.setSizePolicy(sizePolicy)
        self.PretrainedButton.setCheckable(True)
        self.PretrainedButton.setChecked(True)
        self.PretrainedButton.setObjectName("PretrainedButton")
        self.DataLayout.addWidget(self.PretrainedButton)
        spacerItem13 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DataLayout.addItem(spacerItem13)
        self.YourNetworkButton = QPushButton(parent=self.DrawPageW)
        if not selfTrainedDataExists:
            self.YourNetworkButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.YourNetworkButton.sizePolicy().hasHeightForWidth())
        self.YourNetworkButton.setSizePolicy(sizePolicy)
        self.YourNetworkButton.setCheckable(True)
        self.YourNetworkButton.setObjectName("YourNetworkButton")
        self.DataLayout.addWidget(self.YourNetworkButton)
        spacerItem14 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DataLayout.addItem(spacerItem14)
        self.DrawPageL.addLayout(self.DataLayout)

        self.DataLabel = QLabel(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(10)
        self.DataLabel.setFont(font)
        self.DataLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.DataLabel.setObjectName("DataLabel")
        self.DrawPageL.addWidget(self.DataLabel)

        spacerItem15 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawPageL.addItem(spacerItem15)

        self.BackExitLayout = QHBoxLayout()
        self.BackExitLayout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.BackExitLayout.setObjectName("BackExitLayout")
        self.BackButtonDraw = QPushButton(parent=self.DrawPageW)
        self.BackButtonDraw.setObjectName("BackButtonDraw")
        self.BackExitLayout.addWidget(self.BackButtonDraw)
        self.ExitButtonDraw = QPushButton(parent=self.DrawPageW)
        self.ExitButtonDraw.setObjectName("ExitButtonDraw")
        self.BackExitLayout.addWidget(self.ExitButtonDraw)
        self.DrawPageL.addLayout(self.BackExitLayout)
        self.verticalLayout_3.addLayout(self.DrawPageL)
        self.stackedWidget.addWidget(self.DrawPageW)

        # Training Page
        self.TrainingPageW = QWidget()
        self.TrainingPageW.setObjectName("TrainingPageW")
        self.verticalLayout_5 = QVBoxLayout(self.TrainingPageW)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.TrainingPageL = QVBoxLayout()
        self.TrainingPageL.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.TrainingPageL.setObjectName("TrainingPageL")

        self.CostPlotWidget = QSvgWidget()
        if not os.path.isfile("Cost/cost_plot.svg"):
            self.CostPlotWidget.load("Images/cost_plot_empty.svg")
        else:
            self.CostPlotWidget.load("Cost/cost_plot.svg")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.CostPlotWidget.setSizePolicy(sizePolicy)
        self.CostPlotWidget.setObjectName("CostPlotWidget")
        self.TrainingPageL.addWidget(self.CostPlotWidget)

        self.TrainingLabel = QLabel(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainingLabel.sizePolicy().hasHeightForWidth())
        self.TrainingLabel.setSizePolicy(sizePolicy)
        self.TrainingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.TrainingLabel.setObjectName("TrainingLabel")
        self.TrainingPageL.addWidget(self.TrainingLabel)

        self.ProgressBar = QProgressBar(parent=self.TrainingPageW)
        self.ProgressBar.setObjectName("ProgressBar")
        self.ProgressBar.setValue(0)
        self.TrainingPageL.addWidget(self.ProgressBar)

        self.CycleLabel = QLabel(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CycleLabel.sizePolicy().hasHeightForWidth())
        self.CycleLabel.setSizePolicy(sizePolicy)
        self.CycleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.CycleLabel.setObjectName("CycleLabel")
        self.TrainingPageL.addWidget(self.CycleLabel)

        spacerItem16 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingPageL.addItem(spacerItem16)

        self.StopStartLayout = QHBoxLayout()
        self.StopStartLayout.setObjectName("StopStartLayout")
        spacerItem22 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StopStartLayout.addItem(spacerItem22)
        self.StopButton = QPushButton(parent=self.DrawPageW)
        self.StopButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StopButton.sizePolicy().hasHeightForWidth())
        self.StopButton.setSizePolicy(sizePolicy)
        self.StopButton.setCheckable(True)
        self.StopButton.setObjectName("StopButton")
        self.StopStartLayout.addWidget(self.StopButton)
        spacerItem23 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem23)
        self.StartButton = QPushButton(parent=self.DrawPageW)
        if not selfTrainedDataExists:
            self.StartButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartButton.sizePolicy().hasHeightForWidth())
        self.StartButton.setSizePolicy(sizePolicy)
        self.StartButton.setCheckable(True)
        self.StartButton.setObjectName("StartButton")
        self.StopStartLayout.addWidget(self.StartButton)
        spacerItem24 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StopStartLayout.addItem(spacerItem24)
        self.TrainingPageL.addLayout(self.StopStartLayout)

        self.InitializeLayout = QHBoxLayout()
        self.InitializeLayout.setObjectName("InitializeLayout")
        spacerItem17 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.InitializeLayout.addItem(spacerItem17)
        self.InitializeButton = QPushButton(parent=self.TrainingPageW)
        if selfTrainedDataExists:
            self.InitializeButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.InitializeButton.sizePolicy().hasHeightForWidth())
        self.InitializeButton.setSizePolicy(sizePolicy)
        self.InitializeButton.setObjectName("InitializeButton")
        self.InitializeLayout.addWidget(self.InitializeButton)
        spacerItem18 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.InitializeLayout.addItem(spacerItem18)
        self.TrainingPageL.addLayout(self.InitializeLayout)

        self.DeleteLayout = QHBoxLayout()
        self.DeleteLayout.setObjectName("DeleteLayout")
        spacerItem25 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DeleteLayout.addItem(spacerItem25)
        self.DeleteButton = QPushButton(parent=self.TrainingPageW)
        if not selfTrainedDataExists:
            self.DeleteButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DeleteButton.sizePolicy().hasHeightForWidth())
        self.DeleteButton.setSizePolicy(sizePolicy)
        self.DeleteButton.setObjectName("DeleteButton")
        self.DeleteLayout.addWidget(self.DeleteButton)
        spacerItem26 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DeleteLayout.addItem(spacerItem26)
        self.TrainingPageL.addLayout(self.DeleteLayout)

        spacerItem19 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingPageL.addItem(spacerItem19)

        self.BackButtonTrainingLayout = QHBoxLayout()
        self.BackButtonTrainingLayout.setObjectName("BackButtonTrainingLayout")
        spacerItem20 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackButtonTrainingLayout.addItem(spacerItem20)
        self.BackButtonTraining = QPushButton(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BackButtonTraining.sizePolicy().hasHeightForWidth())
        self.BackButtonTraining.setSizePolicy(sizePolicy)
        self.BackButtonTraining.setObjectName("BackButtonTraining")
        self.BackButtonTrainingLayout.addWidget(self.BackButtonTraining)

        spacerItem21 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackButtonTrainingLayout.addItem(spacerItem21)

        self.TrainingPageL.addLayout(self.BackButtonTrainingLayout)
        self.verticalLayout_5.addLayout(self.TrainingPageL)
        self.stackedWidget.addWidget(self.TrainingPageW)



        self.verticalLayout.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.ExitButtonDraw.clicked.connect(MainWindow.close) # type: ignore
        self.ExitButtonMain.clicked.connect(MainWindow.close) # type: ignore
        QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Number Neural Network"))
        MainWindow.setWindowIcon(QIcon("Images/neural_icon.png"))
        self.InfoLabel.setText(_translate("MainWindow", "Info text."))
        self.DrawButton.setText(_translate("MainWindow", "Draw"))
        self.TrainingButton.setText(_translate("MainWindow", "Training"))
        self.ExitButtonMain.setText(_translate("MainWindow", "Exit"))
        self.ClearButton.setText(_translate("MainWindow", "Clear"))
        self.GuessButton.setText(_translate("MainWindow", "Guess the number!"))
        self.CanvasInfoLabel.setText(_translate("MainWindow", "Try to draw large and precise. (1 & 9 are <i>problem numbers</i>.)"))
        self.DataLabel.setText(_translate("MainWindow", "Choose between a pretrained network or a network you have trained"))
        self.PretrainedButton.setText(_translate("MainWindow", "Pretrained"))
        self.YourNetworkButton.setText(_translate("MainWindow", "Your Network"))
        self.BackButtonDraw.setText(_translate("MainWindow", "Back"))
        self.ExitButtonDraw.setText(_translate("MainWindow", "Exit"))
        self.TrainingLabel.setText(_translate("MainWindow", "<br>The <b>Cost</b> values above show how your current network <br>" \
                                                            "performs (on some hidden example numbers). These <i>cost <br> values</i> can be seen as a measure for the networks <br>" \
                                                            "mistakes. The lower the value, the less mistakes it makes. <br> <br>" \
                                                            "Start your network with <b>Initialize Randomly</b> for an <br>" \
                                                            "untrained network with random value association, and <br>" \
                                                            "check how it performs on the <b>Draw</b> page. <br>" \
                                                            "<b>Start Training</b> and see how quickly the <i>cost value</i> drops <br>" \
                                                            "after each training cycle and watch the networks growth <br>"
                                                            "in confidence about your drawn numbers. <br> <br>" \
                                                            "<b>Important!</b> Note that only <b>completed</b> training cycles will <br>" \
                                                            "have an effect on your network! <br>"))
        self.CycleLabel.setText(_translate("MainWindow", f"Training Cycles: {self.CycleNum}"))
        self.StopButton.setText(_translate("MainWindow", "Stop Training"))
        if self.CycleNum > 0:
            self.StartButton.setText(_translate("MainWindow", "Continue Training"))
        else:
            self.StartButton.setText(_translate("MainWindow", "Start Training"))
        self.InitializeButton.setText(_translate("MainWindow", "Initialize Randomly"))
        self.DeleteButton.setText(_translate("MainWindow", "Delete Network"))
        self.BackButtonTraining.setText(_translate("MainWindow", "Back"))



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setMinimumSize(450, 850)

        self.alrTrained = True

        PIX_MAX = 255
        self.test = pd.read_csv('MNIST/mnist_test.csv', index_col=0, header=None)
        self.test = self.test/PIX_MAX


        # Page management
        self.ui.DrawButton.clicked.connect(self.DrawButton_Clicked)
        self.ui.TrainingButton.clicked.connect(self.TrainingButton_Clicked)
        self.ui.BackButtonDraw.clicked.connect(self.BackButton_Clicked)
        self.ui.BackButtonTraining.clicked.connect(self.BackButton_Clicked)

        # Clear & Guess Button
        self.ui.ClearButton.clicked.connect(self.ClearButton_Clicked)
        self.ui.GuessButton.clicked.connect(self.GuessButton_Clicked)

        # Pretrained & Your Network Button
        self.ui.PretrainedButton.clicked.connect(self.PretrainedButton_Clicked)
        self.ui.YourNetworkButton.clicked.connect(self.YourNetworkButton_Clicked)

        self.PretrainedYourNetworkButtonGroup = QButtonGroup()
        self.PretrainedYourNetworkButtonGroup.setExclusive(True)
        self.PretrainedYourNetworkButtonGroup.addButton(self.ui.PretrainedButton)
        self.PretrainedYourNetworkButtonGroup.addButton(self.ui.YourNetworkButton)

        # Stop & Start Training Button
        self.ui.StopButton.clicked.connect(self.StopButton_Clicked)
        self.ui.StartButton.clicked.connect(self.StartButton_Clicked)

        self.StopStartButtonGroup = QButtonGroup()
        self.StopStartButtonGroup.setExclusive(True)
        self.StopStartButtonGroup.addButton(self.ui.StopButton)
        self.StopStartButtonGroup.addButton(self.ui.StartButton)

        # Initialize & Delete Button
        self.ui.InitializeButton.clicked.connect(self.InitializeButton_Clicked)
        self.ui.DeleteButton.clicked.connect(self.DeleteButton_Clicked)


    # Page management
    def DrawButton_Clicked(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def TrainingButton_Clicked(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def BackButton_Clicked(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    # Clear & Guess Button
    def ClearButton_Clicked(self):
        self.ui.Canvas.clearAll()
        self.ui.ResultLabel.setText("")
        self.ui.BarChart.updateValues([0] * 10)
        
    def GuessButton_Clicked(self):
        pixels = self.ui.Canvas.getPixels()
        guessed_num, perc_lst = getGuess(pixels, self.alrTrained)
        self.ui.ResultLabel.setText("Guess: " + str(guessed_num))
        self.ui.BarChart.updateValues(perc_lst)
    
    # Pretrained & Your Network Button
    def PretrainedButton_Clicked(self):
        self.alrTrained = True

    def YourNetworkButton_Clicked(self):
        self.alrTrained = False

    # Stop & Start Training Button
    def StartButton_Clicked(self):
        self.ui.StopButton.setEnabled(True)
        self.ActiveTraining = {"Active": True}

        self.trainingThread = Thread(target=self.trainInThread, daemon=True)
        self.trainingThread.start()

    def trainInThread(self):
        while True:
            if not self.ActiveTraining["Active"]:
                return

            def updateProgress(percentage):
                self.ui.ProgressBar.setValue(int(percentage))
                self.ui.CycleLabel.setText("Training Cycles: " + str(self.ui.CycleNum))

            training(self.test, self.ActiveTraining, progress_callback=updateProgress)
            
            self.ui.CostPlotWidget.load("Cost/cost_plot.svg")
            if self.ui.ProgressBar.value() >= 99:
                self.ui.CycleNum += 1
                with open("WeightsBiases/cycles.json", "w", encoding="utf-8") as file:
                    json.dump({"cycles": self.ui.CycleNum}, file)

    def StopButton_Clicked(self):
        self.ActiveTraining["Active"] = False
        self.ui.ProgressBar.setValue(0)
        if self.ui.CycleNum > 0:
            self.ui.StartButton.setText("Continue Training")

    def InitializeButton_Clicked(self):
        makeRandomWeightsBiases()
        self.ui.StartButton.setEnabled(True)
        self.ui.DeleteButton.setEnabled(True)
        self.ui.InitializeButton.setEnabled(False)
        self.ui.YourNetworkButton.setEnabled(True)


    def DeleteButton_Clicked(self):
        if os.path.exists("WeightsBiases"):
            shutil.rmtree("WeightsBiases")
        if os.path.isfile("Cost/cost.txt"):
            os.remove("Cost/cost.txt")
        if os.path.isfile("Cost/cost_plot.svg"):
            os.remove("Cost/cost_plot.svg")
        self.ui.CostPlotWidget.load("Images/cost_plot_empty.svg")

        self.ui.ProgressBar.setValue(0)
        self.ui.CycleNum = 0
        self.ui.CycleLabel.setText("Training Cycles: " + str(self.ui.CycleNum))
        self.ui.StartButton.setText("Start Training")

        self.PretrainedButton_Clicked()
        
        self.ui.StopButton.setChecked(False)
        self.ui.StartButton.setChecked(False)
        self.ui.PretrainedButton.setChecked(True)
        self.ui.StopButton.setEnabled(False)
        self.ui.StartButton.setEnabled(False)
        self.ui.InitializeButton.setEnabled(True)
        self.ui.YourNetworkButton.setEnabled(False)
        self.ui.DeleteButton.setEnabled(False)
        


        


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


    #TODO: only necessary packages
    #TODO: Delete cost_plot.py and in DeleteButton remove Cost folder
