import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QWidget
from PyQt5.uic import loadUi
from scipy.stats import skew

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
            print(file_path)
            self.processCSV(file_path)

    # must apply same preprocessing and segmentation to the csv file
    def preprocess_data(self, df):
        window_size = 25
        df.columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Absolute_Acc']
        df_filled = df.fillna(method='ffill').copy()
        actual_window = min(window_size, len(df_filled))
        for col in ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Absolute_Acc']:
            df_filled[col] = df_filled[col].rolling(window=actual_window, min_periods=1).mean()
        return df_filled

    def segment_data(self, df):
        window_size, sampling_rate = 5, 50
        window_length = window_size * sampling_rate
        return np.array([df.iloc[i:i + window_length].values for i in range(0, len(df) - window_length + 1, window_length)])

    def extract_features(self, segments):
        feature_list = []
        for segment in segments:
            df = pd.DataFrame(segment, columns=["Time", "Acc_X", "Acc_Y", "Acc_Z", "Absolute_Acc"])
            features = []
            for col in ['Acc_X', 'Acc_Y', 'Acc_Z', 'Absolute_Acc']:
                signal = df[col]
                features.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    np.max(signal) - np.min(signal),  # Range
                    np.var(signal),
                    skew(signal),
                    np.sum(signal ** 2),  # Energy
                    np.mean(np.abs(signal))  # Absolute mean
                ])
            feature_list.append(features)
        return np.array(feature_list)

    def processCSV(self, file_path):
        data = pd.read_csv(file_path)
        print("I have copied the data")
        processed_data = self.preprocess_data(data)
        print("I have processed the data")
        segments = self.segment_data(processed_data)
        print("I have segmented the data")
        features = self.extract_features(segments)
        print("I have extracted the features")




app = QApplication(sys.argv)
main = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(main)
widget.setFixedWidth(800)
widget.setFixedHeight(600)
widget.show()
sys.exit(app.exec_())

