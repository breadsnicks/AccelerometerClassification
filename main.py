import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QWidget
from PyQt5.uic import loadUi

import pickle

from pandas.io.xml import preprocess_data
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("gui.ui", self)

        # Window size control
        self.setFixedSize(800, 600)
        self.browse.clicked.connect(self.browseFiles)

        try:
            with open('accelerometer_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Pipeline loaded successfully. Steps:", self.model.named_steps.keys())
        except Exception as e:
            print(f"Pipeline loading failed: {e}")
            self.model = None

        self.canvas = self.Canvas(self)
        self.plotLayout.addWidget(self.canvas)

    class Canvas(FigureCanvas):
        def __init__(self, parent=None):
            self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=200)
            super().__init__(self.fig)
            self.setParent(parent)
            # Adjust font sizes
            self.ax.set_title("Accelerometer Data", fontsize=8)  # Title font size
            self.ax.set_xlabel("Time", fontsize=5)  # X-axis label font size
            self.ax.set_ylabel("Acceleration", fontsize=5)  # Y-axis label font size

            # Adjust tick font sizes
            self.ax.tick_params(axis='both', which='major', labelsize=5)
            self.ax.grid(True)
            self.ax.legend()

        def plot_data(self, df, title):
            self.ax.clear()
            time = range(len(df))
            x = df[:, 1]
            y = df[:, 2]
            z = df[:, 3]

            self.ax.plot(time, x, label="x")
            self.ax.plot(time, y, label="y")
            self.ax.plot(time, z, label="z")
            self.ax.set_title(f"{title} Accelerometer Data", fontsize=8)  # Title font size
            self.ax.set_xlabel("Time", fontsize=5)  # X-axis label font size
            self.ax.set_ylabel("Acceleration", fontsize=5)  # Y-axis label font size
            self.ax.legend()
            self.draw()

    def browseFiles(self):
        file_path, _ =QFileDialog.getOpenFileName(self, 'Open File', '', '(*.csv)')
        if file_path:
            print(file_path)
            title = self.processCSV(file_path)
            self.plot_csv(file_path, title)

    def plot_csv(self, file_path, title):
        try:
            df = pd.read_csv(file_path)
            df = self.preprocess_data(df)
            self.canvas.plot_data(df.iloc[:len(df)].values, title=title)
        except Exception as e:
            if hasattr(self, 'statusLabel'):
                self.statusLabel.setText(f"Error: {str(e)}")
            print(f"Error loading CSV: {e}")

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
        dominant_activity = 'Walking' if walking_count > jumping_count else 'Jumping'
        return dominant_activity

app = QApplication(sys.argv)
main = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(main)
widget.setFixedWidth(800)
widget.setFixedHeight(600)
widget.show()
sys.exit(app.exec_())

