import sys
import argparse

from PyQt5 import QtCore, QtWidgets

from widgets import ProcessorWidget, LogWidget
from processors import AverageTrackPlotter, ComponentsPlotter, LogCollector, MomentumGraph
from drawers import CsvZRDrawer, CsvXYDrawer




class MainWindow(QtWidgets.QWidget):
    def __init__(self, detector_file, lines, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("GSF Debugger")
        
        drawers = [CsvZRDrawer(detector_file), CsvXYDrawer(detector_file)]
        processors = [
            AverageTrackPlotter(drawers),
            ComponentsPlotter(drawers),
            MomentumGraph(),
        ]
        
        logCollector = LogCollector()

        for line in lines:
            logCollector.parse_line(line)
            for processor in processors:
                processor.parse_line_base(line)

        # Assert all have the same step size
        for p in processors:
            print(p.name(), "steps", p.number_steps())
        
        # assert all([processors[0].number_steps() == p.number_steps() for p in processors ])
        steps = min([ p.number_steps() for p in processors ])


        layout = QtWidgets.QVBoxLayout()

        # Step label
        self.label = QtWidgets.QLabel("step: 0", self)
        layout.addWidget(self.label)

        # Slider

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(steps)
        self.slider.valueChanged.connect(self.step_changed)

        def disable():
            for p in self.processor_widgets:
                if hasattr(p, "is_active"):
                    p.is_active = False

        self.slider.sliderPressed.connect(disable)

        def switch(i=None):
            if i is None:
                i = self.tabs.currentIndex()
            for p in self.processor_widgets:
                if hasattr(p, "is_active"):
                    p.is_active = False
            if hasattr(self.processor_widgets[i], "is_active"):
                self.processor_widgets[i].is_active = True
            self.step_changed()

        self.slider.sliderReleased.connect(switch)

        def bwd():
            self.slider.setValue(max(self.slider.value()-1, 0))
            self.step_changed()

        bwdBtn = QtWidgets.QPushButton("-")
        bwdBtn.pressed.connect(bwd)

        def fwd():
            self.slider.setValue(min(self.slider.value()+1, steps))
            self.step_changed()

        fwdBtn = QtWidgets.QPushButton("+")
        fwdBtn.pressed.connect(fwd)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(bwdBtn, 1)
        hlayout.addWidget(self.slider, 18)
        hlayout.addWidget(fwdBtn, 1)

        layout.addLayout(hlayout)

        # Tabs
        self.processor_widgets = []
        self.tabs = QtWidgets.QTabWidget(self)
        for p in processors:
            self.processor_widgets.append(ProcessorWidget(p))
            self.tabs.addTab(self.processor_widgets[-1], p.name())

        self.processor_widgets.append(LogWidget(logCollector))
        self.tabs.addTab(self.processor_widgets[-1], "Log")
        self.tabs.currentChanged.connect(switch)

        layout.addWidget(self.tabs)

        # init
        switch(0)
        self.step_changed()

        # Finalize
        self.setLayout(layout)
        self.show()
        
    def step_changed(self):
        self.label.setText(f"step: {self.slider.value()}")
        for w in self.processor_widgets:
            w.change_step(self.slider.value())


if __name__ == "__main__":
    # Instantiate parser
    parser = argparse.ArgumentParser(description='GSF Debugger')
    parser.add_argument('detector', help="detector description as csv file", type=str)
    parser.add_argument('--logfile', '-f', help="log file (if not given, stdin is read)", type=str)
    args = vars(parser.parse_args())

    if "logfile" in args:
        with open(args["logfile"], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(args["detector"], lines)
    app.exec_()
