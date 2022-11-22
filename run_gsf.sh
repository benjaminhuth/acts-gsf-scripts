#!/bin/bash

source $HOME/Documents/acts_project/acts/build/python/setup.sh
source $HOME/Documents/acts_project/dependencies/Geant4/bin/geant4.sh
source $HOME/Documents/acts_project/dependencies/DD4hep/bin/thisdd4hep_only.sh

$PREFIX python3 gsf.py $@
