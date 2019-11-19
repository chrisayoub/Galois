#include <cuda.h>
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"
#include "kernels/reduce.cuh"
#include "csr_graph.h"
#include "galois/runtime/cuda/DeviceSync.h"
#include "atomic_helpers.h"
#include "mrbc_tree_cuda.cuh"

#define TB_SIZE 256

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);

// type of short path
using ShortPathType = double;

struct CUDA_Context : public CUDA_Context_Common {
	// Field from NodeData: sourceData, array of BCData

	// Array of minDistance
	struct CUDA_Context_Field<uint32_t*> minDistances;
	// Array of shortPathCount
	struct CUDA_Context_Field<ShortPathType*> shortPathCounts;
	// Array of dependencyValue, treat this as atomics! Use atomic_helpers.h
	struct CUDA_Context_Field<float*> dependencyValues;

	// Remaining fields from NodeData

	// Distance map.
	struct CUDA_Context_Field<CUDATree> dTree;
	// Final bc value
	struct CUDA_Context_Field<float> bc;
	// Index that needs to be pulled in a round
	struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context*) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	load_graph_CUDA_common(ctx, g, num_hosts);

	load_graph_CUDA_field(ctx, &ctx->minDistances, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->shortPathCounts, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->dependencyValues, num_hosts);

	load_graph_CUDA_field(ctx, &ctx->dTree, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);

	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->minDistances.data.zero_gpu();
	ctx->shortPathCounts.data.zero_gpu();
	ctx->dependencyValues.data.zero_gpu();

	ctx->dTree.data.zero_gpu();
	ctx->bc.data.zero_gpu();
	ctx->roundIndexToSend.data.zero_gpu();
}

// Macro functions for sync structures

// minDistances

void get_bitset_minDistances_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->minDistances.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->minDistances, begin, end);
}

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx) {
	ctx->minDistances.is_updated.cpu_rd_ptr()->reset();
}

// dependency

void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->dependencyValues.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->dependencyValues, begin, end);
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx) {
	ctx->dependencyValues.is_updated.cpu_rd_ptr()->reset();
}
