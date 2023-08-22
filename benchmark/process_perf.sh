#!/bin/bash

echo "perf script..."
perf script -a -F comm,pid,tid,time,event,ip,sym,dso > out.perf#

echo "stackcollapse-perf.pl..."
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded

echo "c++filt..."
c++filt < out.folded > out.folded.filt

echo "shorten templates..."
cat out.folded.filt | python3 python/shorten_templates.py > out.folded.filt.short

echo "remove namespaces..."
cat out.folded.filt.short | python3 python/remove_namespaces.py > out.folded.filt.short.namespace

echo "make flamegraph..."
#./FlameGraph/flamegraph.pl --reverse --inverted --minwidth=5 --width=1800 out.folded.filt.short > flamegraph.svg
./FlameGraph/flamegraph.pl --minwidth=25 --width=1800 out.folded.filt.short.namespace > flamegraph.svg
