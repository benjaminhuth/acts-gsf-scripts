#!/bin/bash

BENCHMARK=../build/benchmark/gsf_benchmark/gsf_benchmark

ACTS_PROJECT=$HOME/Documents/acts_project
ODD_DIR=$ACTS_PROJECT/acts/thirdparty/OpenDataDetector

export LD_LIBRARY_PATH=$ACTS_PROJECT/acts/build/thirdparty/OpenDataDetector/factory:$LD_LIBRARY_PATH
source $ACTS_PROJECT/dependencies/DD4hep/bin/thisdd4hep.sh

$PREFIX $BENCHMARK --input 10Kparticles --odd $ODD_DIR/xml/OpenDataDetector.xml --matmap $ODD_DIR/config/odd-material-mapping-config.json
