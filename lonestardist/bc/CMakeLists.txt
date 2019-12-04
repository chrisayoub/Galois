cuda_add_executable(test_mrbc_map test_mrbc_map.cu
    OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

distApp(bc_level)
testDistSyncOnlyApp(bc_level rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -singleSource)

distApp(bc_mr)
testDistSyncOnlyApp(bc_mr rmat15all ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -numRoundSources=4096)
