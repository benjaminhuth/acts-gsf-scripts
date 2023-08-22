#!/bin/bash

export ACTS_ROOT=$HOME/Documents/acts_project/acts
source $ACTS_ROOT/build/python/setup.sh

source $HOME/Documents/acts_project/dependencies/DD4hep/bin/thisdd4hep_only.sh
source $HOME/Documents/acts_project/dependencies/Geant4/bin/geant4.sh


export LD_LIBRARY_PATH=$ACTS_ROOT/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH

rm -f perf.data
perf record --call-graph=dwarf -F1000 python3 python/gsf_benchmark.py --input 10Kparticles
#perf record --call-graph=dwarf,33156 -F99 python3 python/gsf_benchmark.py $@
