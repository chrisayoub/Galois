#include "bc_mr_cuda.cuh"

__global__
void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end,
		float * p_bc, galois::gstl::Vector<BCData> * p_sourceData)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;
  const unsigned __kernel_tb_size = TB_SIZE;

  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_bc[src] = 0.0;
      p_sourceData[src].resize(vectorSize);
    }
  }
}

void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  InitializeGraph <<<blocks, threads>>>(
		  ctx->gg, __begin, __end,
		  ctx->bc.data.gpu_wr_ptr(),
		  ctx->sourceData.data.gpu_wr_ptr()
  );
  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx)
{
  InitializeGraph_cuda(0, ctx->gg.nnodes, ctx);
}

