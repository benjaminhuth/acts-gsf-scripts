import sys

from PyQt5 import QtCore, QtWidgets

from widgets import ProcessorWidget
from processors import AverageTrackPlotter
from drawers import CsvZRDrawer, CsvXYDrawer




class MainWindow(QtWidgets.QWidget):
    def __init__(self, detector_file, input_file, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("GSF Debugger")
        
        drawers = [CsvZRDrawer(detector_file), CsvXYDrawer(detector_file)]
        processors = [AverageTrackPlotter(drawers)]
        
        with open(input_file, 'r') as f:            
            for line in f:
                for processor in processors:
                    processor.parse_line_base(line)

        # Assert all have the same step size
        assert all([processors[0].number_steps() == p.number_steps() for p in processors ])

        layout = QtWidgets.QVBoxLayout()

        # Step label
        self.label = QtWidgets.QLabel("step: 0", self)
        layout.addWidget(self.label)

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(processors[0].fwd_steps) + len(processors[0].bwd_steps))
        self.slider.valueChanged.connect(self.step_changed)
        layout.addWidget(self.slider)
        
        # Tabs
        self.processor_widgets = []
        tabs = QtWidgets.QTabWidget(self)
        for p in processors:
            self.processor_widgets.append(ProcessorWidget(p))
            tabs.addTab(self.processor_widgets[-1], p.name())
        
        layout.addWidget(tabs)

        # Finalize
        self.setLayout(layout)
        self.show()
        
    def step_changed(self):
        self.label.setText(f"step: {self.slider.value()}")
        for w in self.processor_widgets:
            w.change_step(self.slider.value())


detector_file = "../detectors.csv"
input_file = "../output.log"


app = QtWidgets.QApplication(sys.argv)
w = MainWindow(detector_file, input_file)
app.exec_()
