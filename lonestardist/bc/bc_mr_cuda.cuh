#include <cuda.h>
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"
#include "kernels/reduce.cuh"
#include "csr_graph.h"

#include <galois/DistGalois.h>
#include "galois/DReducible.h"

#include "mrbc_tree.h"

#define TB_SIZE 256

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);

struct CUDA_Context : public CUDA_Context_Common {
	// TODO replace with something else, such as array instead of Vector
	struct CUDA_Context_Field<galois::gstl::Vector<BCData>> sourceData;
	// distance map. TODO replace so uses CUDA-compatable hashmap
	struct CUDA_Context_Field<MRBCTree> dTree;
	// final bc value
	struct CUDA_Context_Field<float> bc;
	// index that needs to be pulled in a round
	struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

// TODO not needed?
//void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
//	load_graph_CUDA_common(ctx, g, num_hosts);
//	load_graph_CUDA_field(ctx, &ctx->betweeness_centrality, num_hosts);
//	load_graph_CUDA_field(ctx, &ctx->current_length, num_hosts);
//	load_graph_CUDA_field(ctx, &ctx->dependency, num_hosts);
//	load_graph_CUDA_field(ctx, &ctx->num_shortest_paths, num_hosts);
//	reset_CUDA_context(ctx);
//}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->sourceData.data.zero_gpu();
	ctx->dTree.data.zero_gpu();
	ctx->bc.data.zero_gpu();
	ctx->roundIndexToSend.data.zero_gpu();
}
