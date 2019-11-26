#!/bin/bash

# Build
make -j bc_mr

# Execute this command from build directory to do a small test on a single GPU
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=4096
