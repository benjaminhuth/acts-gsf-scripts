#!/bin/bash

for smearing in "1.0" "0.1" "0.01" "0.001" "0.0001" "0.00001"; do
    echo "Smearing: $smearing"
    ./run_gsf.sh --surfaces=10 -n20 --pmin=4 --pmax=4 -c12 --no_plt_show --smearing=$smearing 2>&1 | grep -E "(correlation plots|Hist|\%\))"
done
