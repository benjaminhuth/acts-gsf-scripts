import matplotlib
matplotlib.use('QtAgg')

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ProcessorWidget(QtWidgets.QWidget):
    def __init__(self, processor, parent=None):
        super(ProcessorWidget, self).__init__()
        self.processor = processor
        self.fig, self.axes = processor.get_figure_axes()
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        
        self.is_active = False
        
    def change_step(self, step):
        try:
            for ax in self.axes:
                ax.cla()
        except:
            self.axes.cla()
            
        if self.is_active:
            self.processor.draw(self.fig, self.axes, step)
            self.canvas.draw()



class LogWidget(QtWidgets.QTextEdit):
    def __init__(self, log_processor, parent=None):
        super(LogWidget, self).__init__()
        self.logs = log_processor.loglines[1:]
        
        font = QtGui.QFont()
        font.setFamily("monospace [Consolas]")
        font.setFixedPitch(True)
        font.setStyleHint(QtGui.QFont.TypeWriter)
        
        self.setFont(font)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        
    def change_step(self, step):
        if step >= len(self.logs):
            self.setText("Error")
        else:
            self.setText("".join(self.logs[step]))
