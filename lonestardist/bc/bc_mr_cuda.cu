#include "bc_mr_cuda.cuh"

// *************************
// ** Kernels (device code)
// *************************

__global__
void InitializeIteration(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		uint32_t** p_minDistances,
		ShortPathType** p_shortPathCounts,
		float** p_dependencyValues,
		uint64_t* nodesToConsider, unsigned numSourcesPerRound)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  for (index_type src = __begin + tid; src < __end; src += nthreads)
  {
	  p_roundIndexToSend[src] = infinity;
	  CUDATree dTree = p_dTree[src];
	  dTree.initialize();

	  uint32_t* minDistances = p_minDistances[src];
	  ShortPathType* shortPathCounts = p_shortPathCounts[src];
	  float* dependencyValues = p_dependencyValues[src];
	  // Loop through sources
	  for (unsigned i = 0; i < numSourcesPerRound; i++) {
		  if (nodesToConsider[i] == graph.node_data[src]) {
			  // This is a source node
			  minDistances[i] = 0;
			  shortPathCounts[i] = 1;
			  dependencyValues[i] = 0.0;
			  dTree.setDistance(i, 0);
		  } else {
			  // This is a non-source node
			  minDistances[i] = infinity;
			  shortPathCounts[i] = 0;
			  dependencyValues[i] = 0.0;
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
		uint32_t** p_minDistances,
		DynamicBitset& bitset_minDistances) {

	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
	dga.thread_entry();

	for (index_type src = __begin + tid; src < __end; src += nthreads)
	{
		uint32_t* minDistances = p_minDistances[src];
		uint32_t* roundIndexToSend = &p_roundIndexToSend[src];
		CUDATree dTree = p_dTree[src];

		uint32_t newRoundIndex = dTree.getIndexToSend(roundNumber);
		*roundIndexToSend = newRoundIndex;

		if (newRoundIndex != infinity) {
			if (minDistances[newRoundIndex] != 0) {
				bitset_minDistances.set(src);
			}
			dga.reduce(1);
		} else if (dTree.moreWork()) {
			dga.reduce(1);
		}
	}

	dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(dga_ts);
}

__global__
void ConfirmMessageToSend(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		const uint32_t roundNumber)
{
	  unsigned tid = TID_1D;
	  unsigned nthreads = TOTAL_THREADS_1D;

	  for (index_type src = __begin + tid; src < __end; src += nthreads)
	  {
		  p_roundIndexToSend[src] = infinity;
		  CUDATree dTree = p_dTree[src];
		  if (p_roundIndexToSend[src] != infinity) {
			  dTree.markSent(roundNumber);
		  }
	  }
}

__global__
void SendAPSPMessages(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		HGAccumulator<uint32_t> dga,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		uint32_t** p_minDistances,
		ShortPathType** p_shortPathCounts)
{
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
	dga.thread_entry();

	for (index_type dest = __begin + tid; dest < __end; dest += nthreads)
	{
		// Loop through current node's edges
		index_type edge_start = graph.getFirstEdge(dest);
		index_type edge_end = graph.getFirstEdge(dest + 1);
		for (index_type edge = edge_start; edge < edge_end; edge++)
		{
			index_type src = graph.getDestination(dest, edge);
			uint32_t indexToSend = p_roundIndexToSend[src];

			if (indexToSend != infinity) {
				uint32_t distValue = p_minDistances[src][indexToSend];
				uint32_t newValue = distValue + 1;
			    // Update minDistance vector
				uint32_t oldValue = p_minDistances[dest][indexToSend];

				if (oldValue > newValue) {
					p_minDistances[dest][indexToSend] = newValue;
					p_dTree[dest].setDistance(indexToSend, oldValue, newValue);
					// overwrite short path with this node's shortest path
					p_shortPathCounts[dest][indexToSend] = p_shortPathCounts[src][indexToSend];
				} else if (oldValue == newValue) {
					// add to short path
					p_shortPathCounts[dest][indexToSend] += p_shortPathCounts[src][indexToSend];
				}

				dga.reduce(1);
			}
		}
	}

	dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(dga_ts);
}

// *******************************
// ** Kernel wrappers (host code)
// ********************************

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx, unsigned int vectorSize)
{
	// Init arrays to be to new arrays of size vectorSize
	// Number of nodes * array size for each node
	size_t arraySize = (size_t) (ctx->gg.nnodes * vectorSize);
	ctx->minDistances.data = Shared<uint32_t*>(arraySize);
	ctx->shortPathCounts.data = Shared<ShortPathType*>(arraySize);
	ctx->dependencyValues.data = Shared<float*>(arraySize);

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
			ctx->minDistances.data.gpu_wr_ptr(),
			ctx->shortPathCounts.data.gpu_wr_ptr(),
			ctx->dependencyValues.data.gpu_wr_ptr(),
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
			ctx->minDistances.data.gpu_wr_ptr(),
			*(ctx->minDistances.is_updated.gpu_rd_ptr())
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;

	// Copy back return value
	dga = *(dgaval.cpu_rd_ptr());
}

void ConfirmMessageToSend_allNodes_cuda(struct CUDA_Context* ctx, const uint32_t roundNumber) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	ConfirmMessageToSend <<<blocks, threads>>>(
			ctx->gg, 0, ctx->gg.nnodes,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->dTree.data.gpu_wr_ptr(),
			roundNumber
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void SendAPSPMessages_nodesWithEdges_cuda(struct CUDA_Context* ctx, uint32_t & dga) {
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
	SendAPSPMessages <<<blocks, threads>>>(
			ctx->gg, 0, ctx->numNodesWithEdges,
			_dga,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->dTree.data.gpu_wr_ptr(),
			ctx->minDistances.data.gpu_wr_ptr(),
			ctx->shortPathCounts.data.gpu_wr_ptr()
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;

	// Copy back return value
	dga = *(dgaval.cpu_rd_ptr());
}

