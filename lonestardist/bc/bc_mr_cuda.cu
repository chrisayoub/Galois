#include "bc_mr_cuda.cuh"

__global__
void InitializeIteration(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		BCData** p_sourceData,
		uint64_t* nodesToConsider, unsigned numSourcesPerRound)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  for (index_type src = __begin + tid; src < __end; src += nthreads)
  {
	  p_roundIndexToSend[src] = infinity;
	  CUDATree dTree = p_dTree[src];
	  dTree.initialize();

	  BCData* bcArray = p_sourceData[src];
	  // Loop through sources
	  for (unsigned i = 0; i < numSourcesPerRound; i++) {
		  if (nodesToConsider[i] == graph.node_data[src]) {
			  // This is a source node
			  bcArray[i].minDistance = 0;
			  bcArray[i].shortPathCount = 1;
			  bcArray[i].dependencyValue = 0.0;
			  dTree.setDistance(i, 0);
		  } else {
			  // This is a non-source node
			  bcArray[i].minDistance = infinity;
			  bcArray[i].shortPathCount = 0;
			  bcArray[i].dependencyValue = 0.0;
		  }
	  }
  }
}

__global__
void FindMessageToSync(CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		HGAccumulator<uint32_t> dga,
		const uint32_t roundNumber,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		BCData** p_sourceData) {

	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
	dga.thread_entry();

	for (index_type src = __begin + tid; src < __end; src += nthreads)
	{
		BCData* sourceData = p_sourceData[src];
		uint32_t* roundIndexToSend = &p_roundIndexToSend[src];
		CUDATree dTree = p_dTree[src];

		uint32_t newRoundIndex = dTree.getIndexToSend(roundNumber);
		*roundIndexToSend = newRoundIndex;

		if (newRoundIndex != infinity) {
			if (sourceData[newRoundIndex].minDistance != 0) {
				// TODO pass this as param so can set
//				bitset_minDistances.set(curNode);
			}
			dga.reduce(1);
		} else if (dTree.moreWork()) {
			dga.reduce(1);
		}
	}

	dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(dga_ts);
}

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

void InitializeIteration_allNodes_cuda(struct CUDA_Context* ctx,
		const std::vector<uint64_t>& nodesToConsider,
		unsigned numSourcesPerRound) {
	// Copy source array to GPU
	uint64_t* nodesArr;
	size_t arrSize = nodesToConsider.size() * sizeof(uint64_t);
	cudaMalloc(&nodesArr, arrSize);
	cudaMemcpy(nodesArr, nodesToConsider.data(), arrSize, cudaMemcpyHostToDevice);

	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	InitializeIteration <<<blocks, threads>>>(ctx->gg, 0, ctx->gg.nnodes,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->dTree.data.gpu_wr_ptr(),
			ctx->sourceData.data.gpu_wr_ptr(),
			nodesArr, numSourcesPerRound);

	// Clean up
	cudaFree(nodesArr);
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void FindMessageToSync_allNodes_cuda(struct CUDA_Context* ctx, const uint32_t roundNumber, uint32_t & dga) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Accumulator
	HGAccumulator<uint32_t> _dga;
	Shared<uint32_t> dgaval  = Shared<uint32_t>(1);
	*(dgaval.cpu_wr_ptr()) = 0;
	_dga.rv = dgaval.gpu_wr_ptr();

	// Kernel call
	FindMessageToSync <<<blocks, threads>>>(
			ctx->gg, 0, ctx->gg.nnodes,
			_dga, roundNumber,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->dTree.data.gpu_wr_ptr(),
			ctx->sourceData.data.gpu_wr_ptr());

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;

	// Copy back return value
	dga = *(dgaval.cpu_rd_ptr());
}
