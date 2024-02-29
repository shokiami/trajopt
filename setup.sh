#!/bin/bash

# clone epigraph
git clone git@github.com:EmbersArc/Epigraph.git

# run cmake
mkdir Epigraph/build
cmake -S Epigraph -B Epigraph/build -DENABLE_ECOS=TRUE -DENABLE_OSQP=TRUE
cmake --build Epigraph/build

# set up lib dir
mkdir lib
cp Epigraph/build/libepigraph.so lib
cp Epigraph/build/solvers/ecos/libecos.so lib
cp Epigraph/build/solvers/osqp/out/libosqp.so lib

# set up obj dir
mkdir obj
