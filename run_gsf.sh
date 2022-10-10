#!/bin/bash

source $HOME/Documents/acts_project/acts/build/python/setup.sh
source $HOME/Documents/acts_project/dependencies/Geant4/bin/geant4.sh

python3 gsf.py $@
