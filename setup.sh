#!/bin/bash
git clone git@github.com:EmbersArc/Epigraph.git
cd Epigraph
mkdir build
cmake --build build
cd ..
cp Epigraph/build/libepigraph.so libs
cp Epigraph/build/solvers/ecos/libecos.so libs
cp Epigraph/build/solvers/osqp/out/libosqp.so libs
export LD_LIBRARY_PATH=libs
