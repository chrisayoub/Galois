#!/bin/bash

# Get inputs
make input

# Build
make -j bc_mr

# Execute this command from build directory to do a small test on a single GPU
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=1024
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat15.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat15.tgr -pset=g -numRoundSources=4096

# Execute this command if on the server
lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat10.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=1024

lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat15.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat15.tgr -pset=g -numRoundSources=4096

# Bigger graph test, start with 1 source and make larger

lonestardist/bc/bc_mr \
 /net/ohm/export/iss/inputs/unweighted/withRandomWeights/livejournal.wgr \ 
 -graphTranspose=/net/ohm/export/iss/inputs/unweighted/withRandomWeights/transpose/livejournal.twgr \
  -pset=g -numRoundSources=1

# TEST CMD
lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat10.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat10.tgr -numRoundSources=1024 -numRuns=10 -pset=g

