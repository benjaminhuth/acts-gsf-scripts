#!/bin/bash

export ACTS_ROOT=$HOME/Documents/acts_project/acts
source $ACTS_ROOT/build/python/setup.sh

source $HOME/Documents/acts_project/dependencies/DD4hep/bin/thisdd4hep_only.sh
source $HOME/Documents/acts_project/dependencies/Geant4/bin/geant4.sh

export LD_LIBRARY_PATH=$ACTS_ROOT/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH

export LD_PRELOAD="/usr/lib64/libprofiler.so"
export CPUPROFILE="cpuprofile.prof"

rm cpuprofile.prof
python3 gsf_benchmark.py $@
