import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QWidget
from PyQt5.uic import loadUi

import pandas as pd
import numpy as np

# Read the ROC data file
roc_data = pd.read_csv('roc_train_data.csv')
fpr = roc_data["False Positive Rate (FPR)"].values
tpr = roc_data["True Positive Rate (TPR)"].values
threshold = roc_data["Thresholds"].values

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
        time = data.iloc[:,0].values
        x = data.iloc[:,1].values
        y = data.iloc[:,2].values
        z = data.iloc[:,3].values
        absolute_acceleration = data.iloc[:,4].values

app = QApplication(sys.argv)
main = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(main)
widget.setFixedWidth(400)
widget.setFixedHeight(300)
widget.show()
sys.exit(app.exec_())

