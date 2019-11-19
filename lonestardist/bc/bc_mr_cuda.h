#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx, unsigned int vectorSize);
void InitializeIteration_allNodes_cuda(struct CUDA_Context* ctx, const std::vector<uint64_t>& nodesToConsider,
		unsigned numSourcesPerRound);
void FindMessageToSync_allNodes_cuda(struct CUDA_Context* ctx, const uint32_t roundNumber, uint32_t & dga);
void ConfirmMessageToSend_allNodes_cuda(struct CUDA_Context* ctx, const uint32_t roundNumber);
void SendAPSPMessages_nodesWithEdges_cuda(struct CUDA_Context* ctx, uint32_t & dga);

// Macros for sync structures
void get_bitset_minDistances_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);

void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
