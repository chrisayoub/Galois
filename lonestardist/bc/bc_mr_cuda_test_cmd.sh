#!/bin/bash

# Build
make -j bc_mr

# Execute this command from build directory to do a small test on a single GPU
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -pset=g -numRoundSources=4096
