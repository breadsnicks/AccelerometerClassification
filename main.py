import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QWidget
from PyQt5.uic import loadUi
from scipy.stats import skew

import pickle

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("gui.ui", self)
        self.browse.clicked.connect(self.browseFiles)
        try:
            with open('accelerometer_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Pipeline loaded successfully. Steps:", self.model.named_steps.keys())
            if hasattr(self.model, 'n_features_in_'):
                print("Expected input features:", self.model.n_features_in_)
        except Exception as e:
            print(f"Pipeline loading failed: {e}")
            self.model = None

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


    def processCSV(self, file_path):
        data = pd.read_csv(file_path)
        print("I have copied the data")
        processed_data = self.preprocess_data(data)
        print("I have processed the data")
        segments = self.segment_data(processed_data)
        print("I have segmented the data")
        if len(segments.shape) == 3:
            n_segments = segments.shape[0]
            flattened_segments = segments.reshape(n_segments, -1)

        predictions = self.model.predict(flattened_segments)
        print("I have predicted the data")
        print(predictions)

        activity_labels = ['walking', 'jumping']
        human_readable = [activity_labels[pred] for pred in predictions]

        # Display results
        print("Predicted activities:", human_readable)
        if hasattr(self, 'resultLabel'):  # If you have a QLabel for results
            self.resultLabel.setText(f"Activities: {', '.join(human_readable)}")

        walking_count = list(predictions).count(0)
        jumping_count = list(predictions).count(1)
        dominant_activity = 'walking' if walking_count > jumping_count else 'jumping'
        print("I think this is", dominant_activity, "data!")

app = QApplication(sys.argv)
main = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(main)
widget.setFixedWidth(800)
widget.setFixedHeight(600)
widget.show()
sys.exit(app.exec_())

