#!/bin/bash

# Get inputs
make input

# Build
make -j bc_mr

# Execute this command from build directory to do a small test on a single GPU
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=4096


# Execute this command if on the server
lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat10.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=4096

