#!/bin/bash

export ACTS_ROOT=$HOME/Documents/acts_project/acts

source $HOME/Documents/acts_project/setup.sh

#python3 make_data.py -n1 -o 10Kparticles --multiplicity=10000 --pmin=1 --pmax=10
python3 make_data.py -n1 -o 10Kparticles --multiplicity=10000 --pmin=1 --pmax=100
