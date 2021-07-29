from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *

from UI.trackingObjectItem import Ui_Form


class toItemWidget(QWidget):
    def __init__(self, parent=None):
        super(toItemWidget, self).__init__(parent)
        self.ui = Ui_Form()

    def sizeHint(self):
        return QSize(239, 145)

    def setId(self, id):
        self.ui.idLabel.setText(id)
    
    def setImg(self, img):
        self.ui.imgLabel.setScaledContents(True)
        self.ui.imgLabel.setPixmap(QPixmap.fromImage(img))

