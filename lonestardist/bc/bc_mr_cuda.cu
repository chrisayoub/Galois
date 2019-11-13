#include "bc_mr_cuda.cuh"

__global__
void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end,
		float * p_bc)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;
  // const unsigned __kernel_tb_size = TB_SIZE;

  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
      p_bc[src] = 0.0;
  }
}

void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx,
		unsigned int vectorSize)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  InitializeGraph <<<blocks, threads>>>(
		  ctx->gg, __begin, __end,
		  ctx->bc.data.gpu_wr_ptr()
  );

  // Init sourceData array to new array of size vectorSize
  // Number of nodes * array size for each node
  ctx->sourceData.data = Shared<BCData>((size_t) ((__end - __begin) * vectorSize));
  // Ensure accessible via GPU
  ctx->sourceData.data.gpu_wr_ptr();

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx, unsigned int vectorSize)
{
  InitializeGraph_cuda(0, ctx->gg.nnodes, ctx, vectorSize);
}

