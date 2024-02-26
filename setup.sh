#!/bin/bash

# clone epigraph
git clone git@github.com:EmbersArc/Epigraph.git

# run cmake
mkdir Epigraph/build
cmake -S Epigraph -B Epigraph/build -DENABLE_ECOS=TRUE -DENABLE_OSQP=True
cmake --build Epigraph/build

# copy libraries
cp Epigraph/build/libepigraph.so libs
cp Epigraph/build/solvers/ecos/libecos.so libs
cp Epigraph/build/solvers/osqp/out/libosqp.so libs
export LD_LIBRARY_PATH=libs
