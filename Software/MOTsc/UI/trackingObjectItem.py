# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'trackingObjectItem.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(239, 145)
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 9, 221, 121))
        self.imgLabel = QLabel(self.groupBox)
        self.imgLabel.setObjectName(u"imgLabel")
        self.imgLabel.setGeometry(QRect(10, 10, 71, 101))
        self.idLabel = QLabel(self.groupBox)
        self.idLabel.setObjectName(u"idLabel")
        self.idLabel.setGeometry(QRect(140, 10, 81, 21))
        font = QFont()
        font.setPointSize(9)
        font.setUnderline(True)
        self.idLabel.setFont(font)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(90, 10, 41, 21))
        font1 = QFont()
        font1.setPointSize(9)
        font1.setBold(True)
        font1.setWeight(75)
        self.label_2.setFont(font1)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox.setTitle("")
        self.imgLabel.setText(QCoreApplication.translate("Form", u"IMG", None))
        self.idLabel.setText(QCoreApplication.translate("Form", u"Uncertain", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"ID:", None))
    # retranslateUi

