import os
import sys
import shutil
import numpy as np
import pandas as pd
import json
from threading import Thread

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QSizePolicy, QSpacerItem, QFrame, QProgressBar, QLayout, QMessageBox, QButtonGroup
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon
from PyQt6.QtCore import Qt, QSize, QMetaObject, QCoreApplication
from PyQt6.QtCharts import QBarSet, QBarSeries, QChart, QChartView, QBarCategoryAxis, QValueAxis
from PyQt6.QtSvgWidgets import QSvgWidget

from neuralnetwork.training import getMNISTData, makeRandomWeightsBiases, getActivations, training
from neuralnetwork.paths import ASSETS_DIR, CURRENT_DIR, MNIST_DIR, TRAINED_DIR



def getGuess(sample, alrTrained):

    if alrTrained:
        w1 = np.array(pd.read_csv(TRAINED_DIR / "w1.csv"))
        b1 = np.array(pd.read_csv(TRAINED_DIR / "b1.csv"))
        w2 = np.array(pd.read_csv(TRAINED_DIR / "w2.csv"))
        b2 = np.array(pd.read_csv(TRAINED_DIR / "b2.csv"))
        w3 = np.array(pd.read_csv(TRAINED_DIR / "w3.csv"))
        b3 = np.array(pd.read_csv(TRAINED_DIR / "b3.csv"))

    else:
        w1 = np.array(pd.read_csv(CURRENT_DIR / "w1.csv"))
        b1 = np.array(pd.read_csv(CURRENT_DIR / "b1.csv"))
        w2 = np.array(pd.read_csv(CURRENT_DIR / "w2.csv"))
        b2 = np.array(pd.read_csv(CURRENT_DIR / "b2.csv"))
        w3 = np.array(pd.read_csv(CURRENT_DIR / "w3.csv"))
        b3 = np.array(pd.read_csv(CURRENT_DIR / "b3.csv"))

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
        self.length = max(1, round(min(self.width(), self.height()) / self.PIXELSIZE))

        grid_size = self.length * self.PIXELSIZE
        x_offset = (self.width() - grid_size) // 2
        y_offset = (self.height() - grid_size) // 2

        idx = 0
        for row in range(self.PIXELSIZE):
            for col in range(self.PIXELSIZE):
                color = QColor("black") if self.pixels[idx] > 0 else QColor("white")
                painter.fillRect(
                    x_offset + col * self.length,
                    y_offset + row * self.length,
                    self.length + 1,
                    self.length + 1,
                    color
                )
                idx += 1


    def mouseMoveEvent(self, event):
        if self.lastPos is not None:
            self.interpolate_paint(self.lastPos, event.position(), event)
        self.lastPos = event.position()


    def mousePressEvent(self, event):
        self.lastPos = event.position()
        self.paint(event)


    def paint(self, event):
        self.length = max(1, round(min(self.width(), self.height()) / self.PIXELSIZE))

        grid_size = self.length * self.PIXELSIZE
        x_offset = (self.width() - grid_size) // 2
        y_offset = (self.height() - grid_size) // 2

        x = int((event.position().x() - x_offset) // self.length)
        y = int((event.position().y() - y_offset) // self.length)

        if 0 <= x < self.PIXELSIZE and 0 <= y < self.PIXELSIZE:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.drawBrush(x, y)
            elif event.buttons() & Qt.MouseButton.RightButton:
                self.pixels[y * self.PIXELSIZE + x] = 0
            self.update()


    def interpolate_paint(self, start_pos, end_pos, event):
        self.length = max(1, round(min(self.width(), self.height()) / self.PIXELSIZE))

        grid_size = self.length * self.PIXELSIZE
        x_offset = (self.width() - grid_size) // 2
        y_offset = (self.height() - grid_size) // 2

        dx = end_pos.x() - start_pos.x()
        dy = end_pos.y() - start_pos.y()
        distance = (dx**2 + dy**2)**0.5

        if distance < 1:
            return

        steps = max(int(distance) + 1, 2)

        for i in range(steps):
            t = i / steps
            px = start_pos.x() + dx * t
            py = start_pos.y() + dy * t

            x = int((px - x_offset) // self.length)
            y = int((py - y_offset) // self.length)

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
        self.barset.setColor(QColor(70, 170, 255))

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(chartview)
        self.setLayout(layout)
    
    def updateValues(self, values):
        for idx, val in enumerate(values):
            self.barset.replace(idx, val)





class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        if os.path.exists(CURRENT_DIR / "cycles.json"):
            with open(CURRENT_DIR / "cycles.json", "r", encoding="utf-8") as file:
                self.CycleNum = json.load(file).get("cycles", 0)
        else:
            self.CycleNum = 0

        self.resolution = QApplication.primaryScreen().size()
        self.res_width = int(self.resolution.width())
        self.res_height = int(self.resolution.height())
        self.target_width = int(self.res_width * 0.28)
        self.target_height = int(self.res_height * 0.89)

        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(int(self.res_width/2 - self.target_width/2), int(self.res_height/2 - self.target_height/2), self.target_width, self.target_height)
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

        selfTrainedDataExists = os.path.exists(CURRENT_DIR) \
                                and os.path.isfile(CURRENT_DIR / "w1.csv") and os.path.isfile(CURRENT_DIR / "b1.csv") \
                                and os.path.isfile(CURRENT_DIR / "w2.csv") and os.path.isfile(CURRENT_DIR / "b2.csv") \
                                and os.path.isfile(CURRENT_DIR / "w3.csv") and os.path.isfile(CURRENT_DIR / "b3.csv")

        # Main Menu
        self.MainMenuW = QWidget()
        self.MainMenuW.setObjectName("MainMenuW")
        self.verticalLayout_2 = QVBoxLayout(self.MainMenuW)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.MainMenuL = QVBoxLayout()
        self.MainMenuL.setSpacing(8)
        self.MainMenuL.setObjectName("MainMenuL")

        self.ImageLayout = QHBoxLayout()
        self.ImageLayout.addStretch()
        self.NetworkImageWidget = QSvgWidget()
        self.NetworkImageWidget.load("src/neuralnetwork/assets/network_image_reduced.svg") # ASSETS_DIR not working.
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.NetworkImageWidget.setSizePolicy(sizePolicy)
        self.NetworkImageWidget.setMinimumWidth(int(self.target_width * 0.75))
        self.NetworkImageWidget.setMinimumHeight(int(self.target_width * 17/24 * 0.75))
        self.NetworkImageWidget.setMaximumWidth(int(self.target_width))
        self.NetworkImageWidget.setMaximumHeight(int(self.target_width * 17/24))
        self.NetworkImageWidget.setObjectName("NetworkImageWidget")
        self.ImageLayout.addWidget(self.NetworkImageWidget)
        self.ImageLayout.addStretch()
        self.MainMenuL.addLayout(self.ImageLayout)

        self.InfoLayout1 = QHBoxLayout()
        self.InfoLayout1.addStretch()
        self.InfoLabel1 = QLabel(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.InfoLabel1.sizePolicy().hasHeightForWidth())
        self.InfoLabel1.setSizePolicy(sizePolicy)
        self.InfoLabel1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.InfoLabel1.setWordWrap(True)
        self.InfoLabel1.setMaximumWidth(int(self.target_width))
        self.InfoLabel1.setObjectName("InfoLabel1")
        self.InfoLayout1.addWidget(self.InfoLabel1)
        self.InfoLayout1.addStretch()
        self.MainMenuL.addLayout(self.InfoLayout1)

        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.MainMenuL.addItem(spacerItem)

        self.InfoLayout2 = QHBoxLayout()
        self.InfoLayout2.addStretch()
        self.InfoLabel2 = QLabel(parent=self.MainMenuW)
        font = QFont()
        font.setPointSize(10)
        self.InfoLabel2.setFont(font)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.InfoLabel2.sizePolicy().hasHeightForWidth())
        self.InfoLabel2.setSizePolicy(sizePolicy)
        self.InfoLabel2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.InfoLabel2.setWordWrap(True)
        self.InfoLabel2.setMinimumWidth(int(self.target_width * 0.8))
        self.InfoLabel2.setMaximumWidth(int(self.target_width))
        self.InfoLabel2.setObjectName("InfoLabel2")
        self.InfoLayout2.addWidget(self.InfoLabel2)
        self.InfoLayout2.addStretch()
        self.MainMenuL.addLayout(self.InfoLayout2)

        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.MainMenuL.addItem(spacerItem)

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
        self.DrawButton.setMaximumWidth(int(self.target_width))
        self.DrawButton.setObjectName("DrawButton")
        self.DrawLayout.addWidget(self.DrawButton)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawLayout.addItem(spacerItem)
        self.MainMenuL.addLayout(self.DrawLayout)

        self.TrainingLayout = QHBoxLayout()
        self.TrainingLayout.setObjectName("TrainingLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingLayout.addItem(spacerItem)
        self.TrainingButton = QPushButton(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainingButton.sizePolicy().hasHeightForWidth())
        self.TrainingButton.setSizePolicy(sizePolicy)
        self.TrainingButton.setMaximumWidth(int(self.target_width))
        self.TrainingButton.setObjectName("TrainingButton")
        self.TrainingLayout.addWidget(self.TrainingButton)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingLayout.addItem(spacerItem)
        self.MainMenuL.addLayout(self.TrainingLayout)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.MainMenuL.addItem(spacerItem)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.ExitButtonMain = QPushButton(parent=self.MainMenuW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ExitButtonMain.sizePolicy().hasHeightForWidth())
        self.ExitButtonMain.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(12)
        self.ExitButtonMain.setFont(font)
        self.ExitButtonMain.setMaximumWidth(int(self.target_width))
        self.ExitButtonMain.setObjectName("ExitButtonMain")
        self.horizontalLayout_2.addWidget(self.ExitButtonMain)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
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

        self.CanvasLayout = QHBoxLayout()
        self.CanvasLayout.setContentsMargins(0, 0, 0, 0)
        self.CanvasLayout.setSpacing(0)
        self.CanvasLayout.addStretch()
        self.Canvas = Canvas()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.Canvas.sizePolicy().hasHeightForWidth())
        self.Canvas.setSizePolicy(sizePolicy)
        self.Canvas.setMaximumWidth(int(self.target_width * 0.9))
        self.Canvas.setBaseSize(QSize(0, 0))
        self.Canvas.setMouseTracking(False)
        self.Canvas.setObjectName("Canvas")
        self.CanvasLayout.addWidget(self.Canvas, 1)
        self.CanvasLayout.addStretch()
        self.DrawPageL.addLayout(self.CanvasLayout, 3)

        self.CanvasInfoLabel = QLabel(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(10)
        self.CanvasInfoLabel.setFont(font)
        self.CanvasInfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.CanvasInfoLabel.setWordWrap(True)
        self.CanvasInfoLabel.setObjectName("CanvasInfoLabel")
        self.DrawPageL.addWidget(self.CanvasInfoLabel)

        spacerItem = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawPageL.addItem(spacerItem)

        self.ClearGuessLayout = QHBoxLayout()
        self.ClearGuessLayout.setObjectName("ClearGuessLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem)
        self.ClearGuessButtonsContainer = QWidget(parent=self.DrawPageW)
        self.ClearGuessButtonsContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ClearGuessButtonsContainer.setMaximumWidth(int(self.target_width))
        self.ClearGuessButtonsLayout = QHBoxLayout(self.ClearGuessButtonsContainer)
        self.ClearGuessButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.ClearGuessButtonsLayout.setSpacing(8)
        self.ClearButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ClearButton.sizePolicy().hasHeightForWidth())
        self.ClearButton.setSizePolicy(sizePolicy)
        self.ClearButton.setObjectName("ClearButton")
        self.ClearGuessButtonsLayout.addWidget(self.ClearButton, 1)
        self.GuessButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GuessButton.sizePolicy().hasHeightForWidth())
        self.GuessButton.setSizePolicy(sizePolicy)
        self.GuessButton.setObjectName("GuessButton")
        self.ClearGuessButtonsLayout.addWidget(self.GuessButton, 1)
        self.ClearGuessLayout.addWidget(self.ClearGuessButtonsContainer)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ClearGuessLayout.addItem(spacerItem)
        self.DrawPageL.addLayout(self.ClearGuessLayout)

        self.ResultLayout = QHBoxLayout()
        self.ResultLayout.setObjectName("ResultLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ResultLayout.addItem(spacerItem)
        self.ResultLabel = QLabel(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ResultLabel.sizePolicy().hasHeightForWidth())
        self.ResultLabel.setSizePolicy(sizePolicy)
        self.ResultLabel.setFrameShape(QFrame.Shape.StyledPanel)
        self.ResultLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ResultLabel.setMaximumWidth(int(self.target_width))
        self.ResultLabel.setObjectName("ResultLabel")
        self.ResultLayout.addWidget(self.ResultLabel)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.ResultLayout.addItem(spacerItem)
        self.DrawPageL.addLayout(self.ResultLayout)

        self.BarChartLayout = QHBoxLayout()
        self.BarChartLayout.setObjectName("BarChartLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BarChartLayout.addItem(spacerItem)
        self.BarChart = ProbabilityBarChart()
        self.BarChart.setObjectName("BarChart")
        self.BarChart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.BarChart.setMaximumWidth(int(self.target_width))
        self.BarChart.setMinimumHeight(int(self.target_height * 0.17))
        self.BarChartLayout.addWidget(self.BarChart)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BarChartLayout.addItem(spacerItem)
        self.DrawPageL.addLayout(self.BarChartLayout)

        self.DataLayout = QHBoxLayout()
        self.DataLayout.setObjectName("DataLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DataLayout.addItem(spacerItem)
        self.DataButtonsContainer = QWidget(parent=self.DrawPageW)
        self.DataButtonsContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.DataButtonsContainer.setMaximumWidth(int(self.target_width))
        self.DataButtonsLayout = QHBoxLayout(self.DataButtonsContainer)
        self.DataButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.DataButtonsLayout.setSpacing(8)
        self.PretrainedButton = QPushButton(parent=self.DrawPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PretrainedButton.sizePolicy().hasHeightForWidth())
        self.PretrainedButton.setSizePolicy(sizePolicy)
        self.PretrainedButton.setCheckable(True)
        self.PretrainedButton.setChecked(True)
        self.PretrainedButton.setObjectName("PretrainedButton")
        self.DataButtonsLayout.addWidget(self.PretrainedButton, 1)
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
        self.DataButtonsLayout.addWidget(self.YourNetworkButton, 1)
        self.DataLayout.addWidget(self.DataButtonsContainer)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DataLayout.addItem(spacerItem)
        self.DrawPageL.addLayout(self.DataLayout)

        self.DataLabel = QLabel(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(10)
        self.DataLabel.setFont(font)
        self.DataLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.DataLabel.setObjectName("DataLabel")
        self.DrawPageL.addWidget(self.DataLabel)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawPageL.addItem(spacerItem)

        self.BackExitLayout = QHBoxLayout()
        self.BackExitLayout.setObjectName("BackExitLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackExitLayout.addItem(spacerItem)
        self.BackExitButtonsContainer = QWidget(parent=self.DrawPageW)
        self.BackExitButtonsContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.BackExitButtonsContainer.setMaximumWidth(int(self.target_width))
        self.BackExitButtonsLayout = QHBoxLayout(self.BackExitButtonsContainer)
        self.BackExitButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.BackExitButtonsLayout.setSpacing(8)
        self.BackButtonDraw = QPushButton(parent=self.DrawPageW)
        self.BackButtonDraw.setObjectName("BackButtonDraw")
        self.BackButtonDraw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.BackExitButtonsLayout.addWidget(self.BackButtonDraw, 1)
        self.ExitButtonDraw = QPushButton(parent=self.DrawPageW)
        self.ExitButtonDraw.setObjectName("ExitButtonDraw")
        self.ExitButtonDraw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.BackExitButtonsLayout.addWidget(self.ExitButtonDraw, 1)
        self.BackExitLayout.addWidget(self.BackExitButtonsContainer)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackExitLayout.addItem(spacerItem)
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

        self.PlotLayout = QHBoxLayout()
        self.PlotLayout.addStretch()
        self.CostPlotWidget = QSvgWidget()
        if not os.path.isfile(ASSETS_DIR / "cost_plot.svg"):
            self.CostPlotWidget.load("src/neuralnetwork/assets/cost_plot_empty.svg") # ASSETS_DIR not working.
        else:
            self.CostPlotWidget.load("src/neuralnetwork/assets/cost_plot.svg") # ASSETS_DIR not working.
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.CostPlotWidget.setSizePolicy(sizePolicy)
        self.CostPlotWidget.setMinimumWidth(int(self.target_width * 0.8))
        self.CostPlotWidget.setMinimumHeight(int(self.target_width * 691/922 * 0.8)) # 922/691: aspect ratio of original plot
        self.CostPlotWidget.setMaximumWidth(int(self.target_width))
        self.CostPlotWidget.setMaximumHeight(int(self.target_width * 691/922))
        self.CostPlotWidget.setObjectName("CostPlotWidget")
        self.PlotLayout.addWidget(self.CostPlotWidget)
        self.PlotLayout.addStretch()
        self.TrainingPageL.addLayout(self.PlotLayout)

        self.TrainingInfoLayout = QHBoxLayout()
        self.TrainingInfoLayout.addStretch()
        self.TrainingLabel = QLabel(parent=self.TrainingPageW)
        font = QFont()
        font.setPointSize(10)
        self.TrainingLabel.setFont(font)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.TrainingLabel.setSizePolicy(sizePolicy)
        self.TrainingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.TrainingLabel.setWordWrap(True)
        self.TrainingLabel.setMaximumWidth(int(self.target_width))
        self.TrainingLabel.setObjectName("TrainingLabel")
        self.TrainingInfoLayout.addWidget(self.TrainingLabel)
        self.TrainingInfoLayout.addStretch()
        self.TrainingPageL.addLayout(self.TrainingInfoLayout)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.TrainingPageL.addItem(spacerItem)

        self.ProgressLayout = QHBoxLayout()
        self.ProgressLayout.addStretch()
        self.ProgressBar = QProgressBar(parent=self.TrainingPageW)
        self.ProgressBar.setObjectName("ProgressBar")
        self.ProgressBar.setValue(0)
        self.ProgressBar.setMinimumWidth(int(self.target_width * 0.9))
        self.ProgressBar.setMaximumWidth(self.target_width)
        self.ProgressLayout.addWidget(self.ProgressBar)
        self.ProgressLayout.addStretch()
        self.TrainingPageL.addLayout(self.ProgressLayout)

        self.CycleLabel = QLabel(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CycleLabel.sizePolicy().hasHeightForWidth())
        self.CycleLabel.setSizePolicy(sizePolicy)
        self.CycleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.CycleLabel.setObjectName("CycleLabel")
        self.TrainingPageL.addWidget(self.CycleLabel)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.TrainingPageL.addItem(spacerItem)

        self.StopStartLayout = QHBoxLayout()
        self.StopStartLayout.setObjectName("StopStartLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StopStartLayout.addItem(spacerItem)
        self.StopStartButtonsContainer = QWidget(parent=self.TrainingPageW)
        self.StopStartButtonsContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.StopStartButtonsContainer.setMaximumWidth(int(self.target_width))
        self.StopStartButtonsLayout = QHBoxLayout(self.StopStartButtonsContainer)
        self.StopStartButtonsLayout.setContentsMargins(0, 0, 0, 0)
        self.StopStartButtonsLayout.setSpacing(self.ClearGuessLayout.spacing())
        self.StopButton = QPushButton(parent=self.TrainingPageW)
        self.StopButton.setEnabled(False)
        self.StopButton.setCheckable(True)
        self.StopButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.StopButton.setMaximumWidth(int((self.target_width - self.StopStartButtonsLayout.spacing()) / 2))
        self.StopButton.setObjectName("StopButton")
        self.StopStartButtonsLayout.addWidget(self.StopButton, 1)
        self.StartButton = QPushButton(parent=self.TrainingPageW)
        if not selfTrainedDataExists:
            self.StartButton.setEnabled(False)
        self.StartButton.setCheckable(True)
        self.StartButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.StartButton.setMaximumWidth(int((self.target_width - self.StopStartButtonsLayout.spacing()) / 2))
        self.StartButton.setObjectName("StartButton")
        self.StopStartButtonsLayout.addWidget(self.StartButton, 1)
        self.StopStartLayout.addWidget(self.StopStartButtonsContainer)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StopStartLayout.addItem(spacerItem)
        self.TrainingPageL.addLayout(self.StopStartLayout)

        self.InitializeLayout = QHBoxLayout()
        self.InitializeLayout.setObjectName("InitializeLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.InitializeLayout.addItem(spacerItem)
        self.InitializeButton = QPushButton(parent=self.TrainingPageW)
        if selfTrainedDataExists:
            self.InitializeButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.InitializeButton.sizePolicy().hasHeightForWidth())
        self.InitializeButton.setSizePolicy(sizePolicy)
        self.InitializeButton.setMaximumWidth(int(self.target_width))
        self.InitializeButton.setObjectName("InitializeButton")
        self.InitializeLayout.addWidget(self.InitializeButton)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.InitializeLayout.addItem(spacerItem)
        self.TrainingPageL.addLayout(self.InitializeLayout)

        self.DeleteLayout = QHBoxLayout()
        self.DeleteLayout.setObjectName("DeleteLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DeleteLayout.addItem(spacerItem)
        self.DeleteButton = QPushButton(parent=self.TrainingPageW)
        if not selfTrainedDataExists:
            self.DeleteButton.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DeleteButton.sizePolicy().hasHeightForWidth())
        self.DeleteButton.setSizePolicy(sizePolicy)
        self.DeleteButton.setMaximumWidth(int(self.target_width))
        self.DeleteButton.setObjectName("DeleteButton")
        self.DeleteLayout.addWidget(self.DeleteButton)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DeleteLayout.addItem(spacerItem)
        self.TrainingPageL.addLayout(self.DeleteLayout)

        spacerItem = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingPageL.addItem(spacerItem)

        self.BackButtonTrainingLayout = QHBoxLayout()
        self.BackButtonTrainingLayout.setObjectName("BackButtonTrainingLayout")
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackButtonTrainingLayout.addItem(spacerItem)
        self.BackButtonTraining = QPushButton(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BackButtonTraining.sizePolicy().hasHeightForWidth())
        self.BackButtonTraining.setSizePolicy(sizePolicy)
        self.BackButtonTraining.setMaximumWidth(int(self.target_width))
        self.BackButtonTraining.setObjectName("BackButtonTraining")
        self.BackButtonTrainingLayout.addWidget(self.BackButtonTraining)

        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.BackButtonTrainingLayout.addItem(spacerItem)

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
        MainWindow.setWindowIcon(QIcon("src/neuralnetwork/assets/neural_icon.png"))
        self.InfoLabel1.setText(_translate("MainWindow", "<b>Welcome to the Number Neural Network!</b> <br><br>" \
                          "This application demonstrates handwritten digit recognition with a feed-forward neural network. <br><br>" \
                          "Use <b>Draw</b> to test predictions in real time with a pretrained model, or open <b>Training</b> to initialize and improve your own model cycle by cycle."))
        self.InfoLabel2.setText(_translate("MainWindow", "The diagram above summarizes the model architecture: 784 input neurons (28x28 pixels with normalized values from 0 to 1), " \
                            "two hidden layers with 16 neurons each, and 10 output neurons representing digits 0-9. " \
                            "Predictions are determined by weights and biases (W and b), which are updated during training to reduce the network's error (<b><i>cost</i></b>). <br>" \
                            "The pretrained model was trained on the <b>MNIST</b> dataset for approximately 60 hours." ))
        self.DrawButton.setText(_translate("MainWindow", "Draw"))
        self.TrainingButton.setText(_translate("MainWindow", "Training"))
        self.ExitButtonMain.setText(_translate("MainWindow", "Exit"))
        self.ClearButton.setText(_translate("MainWindow", "Clear"))
        self.GuessButton.setText(_translate("MainWindow", "Prediction"))
        self.CanvasInfoLabel.setText(_translate("MainWindow", "For best results, draw large and centered digits. <br> (1 and 9 are <i>problem numbers</i>.)"))
        self.DataLabel.setText(_translate("MainWindow", "Select which model to use for prediction."))
        self.PretrainedButton.setText(_translate("MainWindow", "Pretrained Model"))
        self.YourNetworkButton.setText(_translate("MainWindow", "Your Trained Model"))
        self.BackButtonDraw.setText(_translate("MainWindow", "Back"))
        self.ExitButtonDraw.setText(_translate("MainWindow", "Exit"))
        self.TrainingLabel.setText(_translate("MainWindow", "The <b>cost</b> plot above reflects your model's current performance and serves as an indicator of prediction error. " \
                                    "In general, lower cost means fewer mistakes. <br><br>" \
                                    "Use <b>Initialize New Model</b> to create a new untrained model. Select <b>Start Training</b> to run training cycles and then evaluate your network on the <b>Draw</b> page. <br>" \
                                    "Monitor how the cost changes over time. <br><br>" \
                                    "<b>Note:</b> Only <b>completed</b> training cycles are saved and applied to your model."))
        self.CycleLabel.setText(_translate("MainWindow", f"Completed training cycles: {self.CycleNum}"))
        self.StopButton.setText(_translate("MainWindow", "Stop"))
        if self.CycleNum > 0:
            self.StartButton.setText(_translate("MainWindow", "Resume Training"))
        else:
            self.StartButton.setText(_translate("MainWindow", "Start Training"))
        self.InitializeButton.setText(_translate("MainWindow", "Initialize New Model"))
        self.DeleteButton.setText(_translate("MainWindow", "Delete Saved Model"))
        self.BackButtonTraining.setText(_translate("MainWindow", "Back"))



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMinimumSize(int(self.ui.target_width * 0.95), int(self.ui.target_height * 0.985))

        self.alrTrained = True

        if not os.path.isfile(MNIST_DIR / "mnist_test.csv") or not os.path.isfile(MNIST_DIR / "mnist_train.csv"):
            getMNISTData()

        PIX_MAX = 255
        self.test = pd.read_csv(MNIST_DIR / "mnist_test.csv", index_col=0, header=None)
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
        self.ui.DeleteButton.setEnabled(False)

        self.ui.StopButton.setEnabled(True)
        self.ui.StartButton.setEnabled(False)
        self.ActiveTraining = {"Active": True}

        self.trainingThread = Thread(target=self.trainInThread, daemon=True)
        self.trainingThread.start()

    def trainInThread(self):
        while True:
            if not self.ActiveTraining["Active"]:
                return

            def updateProgress(percentage):
                self.ui.ProgressBar.setValue(int(percentage))
                self.ui.CycleLabel.setText("Completed training cycles: " + str(self.ui.CycleNum))

            training(self.test, self.ActiveTraining, progress_callback=updateProgress)
            
            if os.path.isfile(ASSETS_DIR / "cost_plot.svg"):
                self.ui.CostPlotWidget.load("src/neuralnetwork/assets/cost_plot.svg") # ASSETS_DIR not working.

            if self.ui.ProgressBar.value() >= 99:
                self.ui.CycleNum += 1
                with open(CURRENT_DIR / "cycles.json", "w", encoding="utf-8") as file:
                    json.dump({"cycles": self.ui.CycleNum}, file)

    def StopButton_Clicked(self):
        self.ui.StartButton.setEnabled(True)
        self.ui.StopButton.setEnabled(False)
        self.ui.DeleteButton.setEnabled(True)
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

        reply = QMessageBox.question(self, 'Delete Network', 'Are you sure you want to delete your saved model?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return

        if os.path.exists(CURRENT_DIR):
            shutil.rmtree(CURRENT_DIR)
        if os.path.isfile(ASSETS_DIR / "cost_plot.svg"):
            os.remove(ASSETS_DIR / "cost_plot.svg")
        if os.path.isfile(ASSETS_DIR / "cost.txt"):
            os.remove(ASSETS_DIR / "cost.txt")

        self.ui.CostPlotWidget.load("src/neuralnetwork/assets/cost_plot_empty.svg") # ASSETS_DIR not working.

        self.ui.ProgressBar.setValue(0)
        self.ui.CycleNum = 0
        self.ui.CycleLabel.setText("Completed training cycles: " + str(self.ui.CycleNum))
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
    with open(ASSETS_DIR / "style.qss", "r", encoding="utf-8") as file:
        app.setStyleSheet(file.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
