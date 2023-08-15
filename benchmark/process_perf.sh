#!/bin/bash

#echo "perf script..."
#perf script -a -F comm,pid,tid,time,event,ip,sym,dso > out.perf

echo "stackcollapse-perf.pl..."
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded

echo "c++filt..."
c++filt < out.folded > out.folded.filt

echo "shorten templates..."
cat out.folded.filt | ./shorten_templates.py > out.folded.filt.short

echo "make flamegraph..."
./FlameGraph/flamegraph.pl --reverse --inverted --minwidth=5 --width=1800 out.folded.filt.short > flamegraph.svg
