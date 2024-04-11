#!/bin/bash

# helper dirs
mkdir lib
mkdir obj
mkdir data

# epigraph
git clone https://github.com/EmbersArc/Epigraph.git
mkdir Epigraph/build
cmake -S Epigraph -B Epigraph/build -DENABLE_ECOS=TRUE -DENABLE_OSQP=TRUE

if [[ "$(uname)" == "Linux" ]]; then
  # linux
  cmake --build Epigraph/build
  cp Epigraph/build/libepigraph.so lib
  cp Epigraph/build/solvers/ecos/libecos.so lib
  cp Epigraph/build/solvers/osqp/out/libosqp.so lib

elif [[ "$(uname)" == "Darwin" ]]; then
  # macos
  sed -i '' '1s/^/#include <string.h>\n/' Epigraph/solvers/ecos/ecos_bb/ecos_bb.c
  cmake --build Epigraph/build
  cp Epigraph/build/libepigraph.dylib lib
  cp Epigraph/build/solvers/ecos/libecos.dylib lib
  cp Epigraph/build/solvers/osqp/out/libosqp.dylib lib

else
  echo "unsupported operating system"
fi
