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
	// Needed so we can init the correct sizing on the 3 fields below
	unsigned vectorSize;

	// Need to save this value for later
	unsigned num_hosts;

	// Field from NodeData: sourceData, array of BCData

	// Array of minDistance
	struct CUDA_Context_Field<uint32_t> minDistances;
	// Array of shortPathCount
	struct CUDA_Context_Field<ShortPathType> shortPathCounts;
	// Array of dependencyValue, treat this as atomics! Use atomic_helpers.h
	struct CUDA_Context_Field<float> dependencyValues;

	// Remaining fields from NodeData

	// Distance map.
	struct CUDA_Context_Field<CUDATree> dTree;
	// Final bc value
	struct CUDA_Context_Field<float> bc;
	// Index that needs to be pulled in a round
	struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};


// Used to free device mallocs
__global__
void FreeTrees(
		unsigned begin, unsigned int end,
		CUDATree* p_dTree) {
	  unsigned tid = TID_1D;
	  unsigned nthreads = TOTAL_THREADS_1D;
	  for (index_type src = begin + tid; src < end; src += nthreads) {
		  p_dTree[src].dealloc();
	  }
}

void FreeTrees_allNodes(struct CUDA_Context* ctx) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	FreeTrees<<<blocks, threads>>>(0, ctx->gg.nnodes, ctx->dTree.data.gpu_wr_ptr());

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;
}


// CUDA allocations

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context*) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

// Modified from include/galois/cuda/Context.h
template <typename Type>
void load_array_field_CUDA(struct CUDA_Context* ctx, struct CUDA_Context_Field<Type>* field, unsigned num_hosts) {
	// Allocate number of members for data that we needed for flat 2D array
	unsigned data_nmeb = ctx->gg.nnodes * ctx->vectorSize;
	field->data.alloc(data_nmeb);

	// Rest of function stays the same
	size_t max_shared_size = 0; // for union across master/mirror of all hosts
	for (uint32_t h = 0; h < num_hosts; ++h) {
		if (ctx->master.num_nodes[h] > max_shared_size) {
			max_shared_size = ctx->master.num_nodes[h];
		}
	}
	for (uint32_t h = 0; h < num_hosts; ++h) {
		if (ctx->mirror.num_nodes[h] > max_shared_size) {
			max_shared_size = ctx->mirror.num_nodes[h];
		}
	}
	field->shared_data.alloc(max_shared_size);
	field->is_updated.alloc(1);
	field->is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes);
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	load_graph_CUDA_common(ctx, g, num_hosts);

	// Note: not loading fields for sourceData here, need vectorSize later
	// See: InitializeGraph_allNodes_cuda
	// Will save value for later
	ctx->num_hosts = num_hosts;

	load_graph_CUDA_field(ctx, &ctx->dTree, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	// Need to free all mallocs from inside device
	// Must do this before we reset CUDA context
	if (ctx->dTree.data.size() != 0) {
		FreeTrees_allNodes(ctx);
	}

	ctx->minDistances.data.zero_gpu();
	ctx->shortPathCounts.data.zero_gpu();
	ctx->dependencyValues.data.zero_gpu();

	ctx->dTree.data.zero_gpu();
	ctx->bc.data.zero_gpu();
	ctx->roundIndexToSend.data.zero_gpu();
}


float get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float *betweeness_centrality = ctx->bc.data.cpu_rd_ptr();
	return betweeness_centrality[LID];
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
