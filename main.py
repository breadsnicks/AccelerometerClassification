import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QWidget
from PyQt5.uic import loadUi

from joblib import load

import pandas as pd
import numpy as np

# read imported model via joblib
model = load('accelerometer_model.joblib')

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("gui.ui", self)
        self.browse.clicked.connect(self.browseFiles)

    def browseFiles(self):
        file_path, _ =QFileDialog.getOpenFileName(self, 'Open File', '', '(*.csv)')
        if file_path:
            self.processCSV(file_path)

    def processCSV(self, file_path):
        data = pd.read_csv(file_path)
        x = data.iloc[:,1].values
        y = data.iloc[:,2].values
        z = data.iloc[:,3].values

        accel_data = np.column_stack((x, y, z))

app = QApplication(sys.argv)
main = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(main)
widget.setFixedWidth(800)
widget.setFixedHeight(600)
widget.show()
sys.exit(app.exec_())

