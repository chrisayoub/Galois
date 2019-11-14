#include "bc_mr_cuda.cuh"

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx, unsigned int vectorSize)
{
  // Init sourceData array to new array of size vectorSize
  // Number of nodes * array size for each node
  ctx->sourceData.data = Shared<BCData*>((size_t) (ctx->gg.nnodes * vectorSize));

  // Set all memory to 0
  reset_CUDA_context(ctx);

  // Finish op
  cudaDeviceSynchronize();
  check_cuda_kernel;
}

