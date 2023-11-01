import matplotlib
matplotlib.use('QtAgg')

from PyQt5 import QtCore, QtWidgets

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
        
        self.change_step(6)
        
    def change_step(self, step):
        for ax in self.axes: 
            ax.cla()
            
        self.processor.draw(self.fig, self.axes, step)
        self.canvas.draw()
