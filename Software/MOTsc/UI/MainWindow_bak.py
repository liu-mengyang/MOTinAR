# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1247, 657)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionPause = QAction(MainWindow)
        self.actionPause.setObjectName(u"actionPause")
        icon = QIcon()
        icon.addFile(u"pause.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionPause.setIcon(icon)
        self.actionContinue = QAction(MainWindow)
        self.actionContinue.setObjectName(u"actionContinue")
        icon1 = QIcon()
        icon1.addFile(u"continue.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionContinue.setIcon(icon1)
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        icon2 = QIcon()
        icon2.addFile(u"close.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionClose.setIcon(icon2)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.titleLabel = QLabel(self.centralwidget)
        self.titleLabel.setObjectName(u"titleLabel")
        self.titleLabel.setGeometry(QRect(330, 20, 571, 41))
        font = QFont()
        font.setFamily(u"Microsoft YaHei")
        font.setPointSize(18)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(920, 19, 311, 161))
        font1 = QFont()
        font1.setFamily(u"Microsoft YaHei")
        font1.setPointSize(11)
        self.groupBox.setFont(font1)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 30, 151, 31))
        font2 = QFont()
        font2.setPointSize(9)
        font2.setBold(True)
        font2.setWeight(75)
        self.label.setFont(font2)
        self.videotypeLabel = QLabel(self.groupBox)
        self.videotypeLabel.setObjectName(u"videotypeLabel")
        self.videotypeLabel.setGeometry(QRect(160, 30, 141, 31))
        font3 = QFont()
        font3.setPointSize(9)
        font3.setUnderline(True)
        self.videotypeLabel.setFont(font3)
        self.locationurlLabel = QLabel(self.groupBox)
        self.locationurlLabel.setObjectName(u"locationurlLabel")
        self.locationurlLabel.setGeometry(QRect(160, 60, 141, 31))
        self.locationurlLabel.setFont(font3)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 60, 151, 31))
        self.label_2.setFont(font2)
        self.videonameLabel = QLabel(self.groupBox)
        self.videonameLabel.setObjectName(u"videonameLabel")
        self.videonameLabel.setGeometry(QRect(160, 90, 141, 31))
        self.videonameLabel.setFont(font3)
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 90, 151, 31))
        self.label_3.setFont(font2)
        self.resolutionLabel = QLabel(self.groupBox)
        self.resolutionLabel.setObjectName(u"resolutionLabel")
        self.resolutionLabel.setGeometry(QRect(160, 120, 141, 31))
        self.resolutionLabel.setFont(font3)
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 120, 141, 31))
        self.label_4.setFont(font2)
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(920, 180, 311, 101))
        self.groupBox_2.setFont(font1)
        self.trackingmethodLabel = QLabel(self.groupBox_2)
        self.trackingmethodLabel.setObjectName(u"trackingmethodLabel")
        self.trackingmethodLabel.setGeometry(QRect(160, 30, 141, 31))
        self.trackingmethodLabel.setFont(font3)
        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 30, 151, 31))
        self.label_5.setFont(font2)
        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 60, 151, 31))
        self.label_6.setFont(font2)
        self.fpsLabel = QLabel(self.groupBox_2)
        self.fpsLabel.setObjectName(u"fpsLabel")
        self.fpsLabel.setGeometry(QRect(160, 60, 141, 31))
        self.fpsLabel.setFont(font3)
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 410, 291, 151))
        self.groupBox_3.setFont(font1)
        self.outputTextBrowser = QTextBrowser(self.groupBox_3)
        self.outputTextBrowser.setObjectName(u"outputTextBrowser")
        self.outputTextBrowser.setGeometry(QRect(10, 40, 261, 61))
        self.saveLocationLabel = QLabel(self.groupBox_3)
        self.saveLocationLabel.setObjectName(u"saveLocationLabel")
        self.saveLocationLabel.setGeometry(QRect(10, 120, 121, 21))
        font4 = QFont()
        font4.setFamily(u"Microsoft YaHei")
        font4.setPointSize(9)
        self.saveLocationLabel.setFont(font4)
        self.saveLocationLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.saveLocationLineEdit = QLineEdit(self.groupBox_3)
        self.saveLocationLineEdit.setObjectName(u"saveLocationLineEdit")
        self.saveLocationLineEdit.setGeometry(QRect(120, 120, 81, 20))
        font5 = QFont()
        font5.setPointSize(9)
        self.saveLocationLineEdit.setFont(font5)
        self.browsePushButton = QPushButton(self.groupBox_3)
        self.browsePushButton.setObjectName(u"browsePushButton")
        self.browsePushButton.setGeometry(QRect(210, 120, 71, 23))
        self.browsePushButton.setFont(font5)
        self.showTxtRadioButton = QRadioButton(self.groupBox_3)
        self.showTxtRadioButton.setObjectName(u"showTxtRadioButton")
        self.showTxtRadioButton.setGeometry(QRect(10, 20, 81, 21))
        font6 = QFont()
        font6.setFamily(u"Microsoft YaHei")
        font6.setPointSize(9)
        font6.setBold(False)
        font6.setWeight(50)
        self.showTxtRadioButton.setFont(font6)
        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(10, 10, 291, 381))
        self.groupBox_4.setFont(font1)
        self.trackingobjectsList = QListWidget(self.groupBox_4)
        self.trackingobjectsList.setObjectName(u"trackingobjectsList")
        self.trackingobjectsList.setGeometry(QRect(10, 40, 256, 331))
        self.showTORadioButton = QRadioButton(self.groupBox_4)
        self.showTORadioButton.setObjectName(u"showTORadioButton")
        self.showTORadioButton.setGeometry(QRect(10, 20, 81, 21))
        self.showTORadioButton.setFont(font6)
        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(320, 70, 581, 491))
        self.groupBox_5.setFont(font1)
        self.videoLabel = QLabel(self.groupBox_5)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setGeometry(QRect(10, 40, 561, 431))
        self.videoLabel.setFont(font5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1247, 26))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionContinue)
        self.toolBar.addAction(self.actionPause)
        self.toolBar.addAction(self.actionClose)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open..", None))
        self.actionPause.setText(QCoreApplication.translate("MainWindow", u"pause", None))
        self.actionContinue.setText(QCoreApplication.translate("MainWindow", u"continue", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"close", None))
        self.titleLabel.setText(QCoreApplication.translate("MainWindow", u"Realtime Multi-Object Tracking System", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Video Information", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Video type            :", None))
        self.videotypeLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.locationurlLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Location/URL       :", None))
        self.videonameLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Video name          :", None))
        self.resolutionLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Resolution             :", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Processing Information", None))
        self.trackingmethodLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Tracking method :", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"FPS                          :", None))
        self.fpsLabel.setText(QCoreApplication.translate("MainWindow", u"Uncertain", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Trcklets TXT", None))
        self.saveLocationLabel.setText(QCoreApplication.translate("MainWindow", u"Save location\uff1a", None))
        self.browsePushButton.setText(QCoreApplication.translate("MainWindow", u"browse", None))
        self.showTxtRadioButton.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Tracking Objects", None))
        self.showTORadioButton.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Tracking Video", None))
        self.videoLabel.setText(QCoreApplication.translate("MainWindow", u"[Show Postprocessed Video]", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

