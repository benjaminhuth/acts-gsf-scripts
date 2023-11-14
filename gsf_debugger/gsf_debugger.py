#!/bin/env python3

import argparse
import os
import sys
import copy

from PyQt5 import QtCore, QtWidgets

from src.processors import AverageTrackPlotter, ComponentsPlotter, MomentumGraph
from src.drawers import CsvZRDrawer, CsvXYDrawer

from src.main import MainWindow

def groupLogToSteps(lines):
    steps = []
    
    current_step = []
    for line in lines:
        # TODO this is maybe not super reliable and also highly GSF specific
        if line.count("at mean position") == 1:
            steps.append(copy.deepcopy(current_step))
            current_step = []
            
        current_step.append(line)
        
    return steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GSF Debugger')
    parser.add_argument('--detector', '-d', help="detector description as csv file", type=str)
    parser.add_argument('--logfile', '-f', help="log file (if not given, stdin is read)", type=str)
    args = vars(parser.parse_args())
    print(args)
    
    if args["detector"] is None and "detectors.csv" in os.listdir():
        args["detector"] = "detectors.csv"

    if args["logfile"] is not None:
        with open(args["logfile"], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
        
    steps = groupLogToSteps(lines)
    
    
    drawers = [CsvZRDrawer(args["detector"]), CsvXYDrawer(args["detector"])]
    processors = [
        AverageTrackPlotter(drawers),
        ComponentsPlotter(drawers),
        MomentumGraph(),
    ]
    
    for step in steps:
        for processor in processors:
            processor.parse_step(step)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(processors, steps)
    app.exec_()
