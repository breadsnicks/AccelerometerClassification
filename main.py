import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Accelerometer Classification")
        self.setGeometry(700, 300, 800, 600)  # Initial position (x, y, width, height)
        self.setWindowIcon(QIcon("coatofarms.png"))
        self.center_window()

    def center_window(self):
        frame_geometry = self.frameGeometry()  # Get the window's size
        screen_center = QDesktopWidget().availableGeometry().center()  # Get the screen's center
        frame_geometry.moveCenter(screen_center)  # Move the window's center
        self.move(frame_geometry.topLeft())  # Set the new top-left position

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
