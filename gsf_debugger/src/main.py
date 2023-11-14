import sys
import argparse

from PyQt5 import QtCore, QtWidgets

from .widgets import ProcessorWidget, LogWidget

class MainWindow(QtWidgets.QWidget):
    def __init__(self, processors, steps, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("GSF Debugger")

        layout = QtWidgets.QVBoxLayout()

        # Step label
        self.label = QtWidgets.QLabel("step: 0", self)
        layout.addWidget(self.label)

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(steps))
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
            self.slider.setValue(min(self.slider.value()+1, len(steps)))
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

        self.processor_widgets.append(LogWidget(steps))
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

