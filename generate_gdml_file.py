from pathlib import Path
import math
import pprint
import os

import argparse
import numpy as np

from gdml_telescope import *

parser = argparse.ArgumentParser(description='Construct the GDML File for a very simple telescope geometry')
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=5)
parser.add_argument('--surface_distance', help='distance between telescope surfaces', type=int, default=50)
parser.add_argument('--surface_thickness', help='thickness of telescope surfaces', type=int, default=1)
parser.add_argument('--surface_width', help='width of telescope surfaces', type=int, default=1000)
args = vars(parser.parse_args())


surface_distance = args["surface_distance"]
surface_thickness = args["surface_thickness"]
surface_width = args["surface_width"]
x_offset = args["surfaces"] * surface_distance / 2
positions = np.arange(-x_offset, args["surfaces"]*surface_distance, surface_distance).tolist()
gdml_file = "gdml/telescope.gdml"
Path(gdml_file).parent.mkdir(exist_ok=True)
make_gdml(gdml_file, args["surfaces"], surface_width, surface_thickness, surface_distance)