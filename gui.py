import os
import sys
import numpy as np
import pandas as pd

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtCharts import *

from train import getActivations



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
        #self.setFixedSize(QSize(self.window_size, self.window_size))
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




# Qt Designer Code
class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(900, 80, 600, 850)
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
        #self.Canvas = QWidget(parent=self.DrawPageW)
        self.Canvas = Canvas()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.Canvas.sizePolicy().hasHeightForWidth())
        self.Canvas.setSizePolicy(sizePolicy)
        self.Canvas.setMinimumSize(QSize(0, 0))
        self.Canvas.setBaseSize(QSize(0, 0))
        self.Canvas.setMouseTracking(False)
        self.Canvas.setObjectName("Canvas")
        self.DrawPageL.addWidget(self.Canvas)
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
        #self.BarChart = QWidget(parent=self.DrawPageW)
        self.BarChart = ProbabilityBarChart()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.BarChart.sizePolicy().hasHeightForWidth())
        self.BarChart.setSizePolicy(sizePolicy)
        self.BarChart.setSizeIncrement(QSize(0, 0))
        self.BarChart.setObjectName("BarChart")
        self.DrawPageL.addWidget(self.BarChart)
        self.DataLabel = QLabel(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(10)
        self.DataLabel.setFont(font)
        self.DataLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.DataLabel.setObjectName("DataLabel")
        self.DrawPageL.addWidget(self.DataLabel)
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

        selfTrainedDataExists = os.path.exists("WeightsBiases") \
                                and os.path.isfile("WeightsBiases/w1.csv") and os.path.isfile("WeightsBiases/b1.csv") \
                                and os.path.isfile("WeightsBiases/w1.csv") and os.path.isfile("WeightsBiases/b2.csv") \
                                and os.path.isfile("WeightsBiases/w3.csv") and os.path.isfile("WeightsBiases/b3.csv")

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
        spacerItem15 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.DrawPageL.addItem(spacerItem15)
        self.BackExitLayout = QHBoxLayout()
        self.BackExitLayout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.BackExitLayout.setObjectName("BackExitLayout")
        self.BackButtonDraw = QPushButton(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.BackButtonDraw.setFont(font)
        self.BackButtonDraw.setObjectName("BackButtonDraw")
        self.BackExitLayout.addWidget(self.BackButtonDraw)
        self.ExitButtonDraw = QPushButton(parent=self.DrawPageW)
        font = QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.ExitButtonDraw.setFont(font)
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
        self.TrainingLabel = QLabel(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainingLabel.sizePolicy().hasHeightForWidth())
        self.TrainingLabel.setSizePolicy(sizePolicy)
        self.TrainingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.TrainingLabel.setObjectName("TrainingLabel")
        self.TrainingPageL.addWidget(self.TrainingLabel)
        spacerItem16 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.TrainingPageL.addItem(spacerItem16)
        self.StartLayout = QHBoxLayout()
        self.StartLayout.setObjectName("StartLayout")
        spacerItem17 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StartLayout.addItem(spacerItem17)
        self.StartButton = QPushButton(parent=self.TrainingPageW)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartButton.sizePolicy().hasHeightForWidth())
        self.StartButton.setSizePolicy(sizePolicy)
        self.StartButton.setObjectName("StartButton")
        self.StartLayout.addWidget(self.StartButton)
        spacerItem18 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.StartLayout.addItem(spacerItem18)
        self.TrainingPageL.addLayout(self.StartLayout)
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
        self.InfoLabel.setText(_translate("MainWindow", "Info text."))
        self.DrawButton.setText(_translate("MainWindow", "Draw!"))
        self.TrainingButton.setText(_translate("MainWindow", "Training"))
        self.ExitButtonMain.setText(_translate("MainWindow", "Exit"))
        self.ClearButton.setText(_translate("MainWindow", "Clear"))
        self.GuessButton.setText(_translate("MainWindow", "Guess the number!"))
        self.DataLabel.setText(_translate("MainWindow", "Choose between a pretrained network or the network that you have trained"))
        self.PretrainedButton.setText(_translate("MainWindow", "Pretrained"))
        self.YourNetworkButton.setText(_translate("MainWindow", "Your Network"))
        self.BackButtonDraw.setText(_translate("MainWindow", "Back"))
        self.ExitButtonDraw.setText(_translate("MainWindow", "Exit"))
        self.TrainingLabel.setText(_translate("MainWindow", "Training Info"))
        self.StartButton.setText(_translate("MainWindow", "Start Training!"))
        self.BackButtonTraining.setText(_translate("MainWindow", "Back"))



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.alrTrained = True


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

        # Start Training Button
        self.ui.StartButton.clicked.connect(self.StartButton_Clicked)


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

    # Start Training Button
    def StartButton_Clicked(self):
        pass



if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
