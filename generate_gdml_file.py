from pathlib import Path
import math
import pprint
import os

import argparse
import numpy as np

from gdml_telescope import *

parser = argparse.ArgumentParser(description='Construct the GDML File for a very simple telescope geometry')
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=5)
args = vars(parser.parse_args())


surface_distance = 50
surface_thickness = 1
surface_width = 1000
x_offset = args["surfaces"] * surface_distance / 2
positions = np.arange(-x_offset, args["surfaces"]*surface_distance, surface_distance).tolist()
gdml_file = "gdml/telescope.gdml"
Path(gdml_file).parent.mkdir(exist_ok=True)
make_gdml(gdml_file, args["surfaces"], surface_width, surface_thickness, surface_distance)