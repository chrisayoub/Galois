#!/bin/bash

# Get inputs
make input

# Build
make -j bc_mr

# Execute this command from build directory to do a small test on a single GPU
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=1024
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat15.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat15.tgr -pset=g -numRoundSources=4096 -runs=1

# Execute this command if on the server
lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat10.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat10.tgr -pset=g -numRoundSources=1024

lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat15.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat15.tgr -pset=g -numRoundSources=4096

# GPU cmd local
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr  -numRoundSources=1024 -runs=10 -pset=g | grep Timer_

# CPU cmd local
lonestardist/bc/bc_mr inputs/small_inputs/scalefree/rmat10.gr -graphTranspose=inputs/small_inputs/scalefree/transpose/rmat10.tgr  -numRoundSources=1024 -runs=10 -pset=c -t=4 | grep Timer_

# cmake
cmake ../. -DENABLE_HETERO_GALOIS=1  # -DCMAKE_BUILD_TYPE=Debug 

############################################################
############################################################
# IGNORE ABOVE STUFF 
############################################################

# Tester bc_level
lonestardist/bc/bc_level /net/ohm/export/iss/inputs/scalefree/rmat15.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat15.tgr -pset=g -runs=10 -statFile=level.txt > level_verify.txt
 
# Tester bc_mr
lonestardist/bc/bc_mr /net/ohm/export/iss/inputs/scalefree/rmat15.gr -graphTranspose=/net/ohm/export/iss/inputs/scalefree/transpose/rmat15.tgr -pset=g -runs=10 -numRoundSources=1024 -statFile=mr.txt > mr_verify.txt

# Compare times
grep Timer_ level.txt
grep Timer_  mr.txt 

# Verify vals
grep 'BC ' level_verify.txt 
grep 'BC '  mr_verify.txt 


