cuda_add_executable(test_mrbc_map test_mrbc_map.cu
    OPTIONS -g -G ${GENCODE} ${CUDA_VERBOSE_PTXAS})
target_link_libraries(test_mrbc_map galois_gpu)

distApp(bc_level)
testDistSyncOnlyApp(bc_level rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -singleSource)

distApp(bc_mr)
testDistSyncOnlyApp(bc_mr rmat15all ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -numRoundSources=4096)
