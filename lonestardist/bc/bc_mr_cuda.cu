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

__global__
void RoundUpdate(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		CUDATree* p_dTree)
{
	  unsigned tid = TID_1D;
	  unsigned nthreads = TOTAL_THREADS_1D;

	  for (index_type src = __begin + tid; src < __end; src += nthreads)
	  {
		  p_dTree[src].prepForBackPhase();
	  }
}

__global__
void BackFindMessageToSend(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		const uint32_t roundNumber,
        const uint32_t lastRoundNumber,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		float** p_dependencyValues,
		DynamicBitset& bitset_dependency)
{
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type src = __begin + tid; src < __end; src += nthreads) {
		// if zero distances already reached, there is no point sending things
		// out since we don't care about dependecy for sources (i.e. distance
		// 0)
		CUDATree dTree = p_dTree[src];
		if (!dTree.isZeroReached()) {
			uint32_t newRoundIndex = dTree.backGetIndexToSend(roundNumber, lastRoundNumber);
			p_roundIndexToSend[src] = newRoundIndex;

			if (newRoundIndex != infinity) {
	            // only comm if not redundant 0
				if (p_dependencyValues[src][newRoundIndex] != 0) {
					bitset_dependency.set(src);
				}
			}
		}
	}
}

__global__
void BackProp(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		uint32_t* p_roundIndexToSend,
		uint32_t** p_minDistances,
		ShortPathType** p_shortPathCounts,
		float** p_dependencyValues) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type dest = __begin + tid; dest < __end; dest += nthreads) {
		unsigned i = p_roundIndexToSend[dest];

		if (i != infinity) {
			uint32_t myDistance = p_minDistances[dest][i];

		    // calculate final dependency value
			p_dependencyValues[dest][i] = p_dependencyValues[dest][i] * p_shortPathCounts[dest][i];

			// get the value to add to predecessors
			float toAdd = ((float)1 + p_dependencyValues[dest][i]) /
					p_shortPathCounts[dest][i];

			// Loop through current node's edges
			index_type edge_start = graph.getFirstEdge(dest);
			index_type edge_end = graph.getFirstEdge(dest + 1);
			for (index_type edge = edge_start; edge < edge_end; edge++)
			{
				index_type src = graph.getDestination(dest, edge);
				uint32_t sourceDistance = p_minDistances[src][i];

				// source nodes of this batch (i.e. distance 0) can be safely
				// ignored
				if (sourceDistance != 0) {
					// determine if this source is a predecessor
					if (myDistance == (sourceDistance + 1)) {
						// add to dependency of predecessor using our finalized one
						atomicTestAdd(&p_dependencyValues[src][i], toAdd);
					}
				}
			}
		}
	}
}

__global__
void BC(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		float* p_bc,
		float** p_dependencyValues,
		uint64_t* nodesToConsider, unsigned numSourcesPerRound) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type src = __begin + tid; src < __end; src += nthreads) {
		for (unsigned i = 0; i < numSourcesPerRound; i++) {
			// exclude sources themselves from BC calculation
			if (graph.node_data[src] != nodesToConsider[i]) {
				p_bc[src] += p_dependencyValues[src][i];
			}
		}
	}
}

__global__
void Sanity(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		float* p_bc,
		HGAccumulator<float> sum,
		HGReduceMax<float> max,
		HGReduceMin<float> min) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_sum_ts;
	__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_max_ts;
	__shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_min_ts;

	sum.thread_entry();
	max.thread_entry();
	min.thread_entry();

	for (index_type src = __begin + tid; src < __end; src += nthreads) {
		float bc = p_bc[src];
		max.reduce(bc);
		min.reduce(bc);
		sum.reduce(bc);
	}

	sum.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_ts);
	max.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_max_ts);
	min.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_min_ts);
}

// *******************************
// ** Helper functions (host code)
// ********************************

uint64_t* copyVectorToDevice(const std::vector<uint64_t>& vec) {
	uint64_t* arr;
	size_t arrSize = vec.size() * sizeof(uint64_t);
	cudaMalloc(&arr, arrSize);
	cudaMemcpy(arr, vec.data(), arrSize, cudaMemcpyHostToDevice);
	return arr;
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
	uint64_t* nodesArr = copyVectorToDevice(nodesToConsider);

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

void RoundUpdate_allNodes_cuda(struct CUDA_Context* ctx) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	RoundUpdate <<<blocks, threads>>>(
			ctx->gg, 0, ctx->gg.nnodes,
			ctx->dTree.data.gpu_wr_ptr()
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void BackFindMessageToSend_allNodes_cuda(struct CUDA_Context* ctx,
		const uint32_t roundNumber,
        const uint32_t lastRoundNumber) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	BackFindMessageToSend <<<blocks, threads>>>(
			ctx->gg, 0, ctx->gg.nnodes,
			roundNumber,
	        lastRoundNumber,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->dTree.data.gpu_wr_ptr(),
			ctx->dependencyValues.data.gpu_wr_ptr(),
			*(ctx->dependencyValues.is_updated.gpu_rd_ptr())
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void BackProp_nodesWithEdges_cuda(struct CUDA_Context* ctx) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	BackProp <<<blocks, threads>>>(
			ctx->gg, 0, ctx->numNodesWithEdges,
			ctx->roundIndexToSend.data.gpu_wr_ptr(),
			ctx->minDistances.data.gpu_wr_ptr(),
			ctx->shortPathCounts.data.gpu_wr_ptr(),
			ctx->dependencyValues.data.gpu_wr_ptr()
	);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void BC_masterNodes_cuda(struct CUDA_Context* ctx,
		const std::vector<uint64_t>& nodesToConsider,
		unsigned numSourcesPerRound) {
	// Copy source array to GPU
	uint64_t* nodesArr = copyVectorToDevice(nodesToConsider);

	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Kernel call
	BC <<<blocks, threads>>>(
			ctx->gg, ctx->beginMaster, ctx->beginMaster + ctx->numOwned,
			ctx->bc.data.gpu_wr_ptr(),
			ctx->dependencyValues.data.gpu_wr_ptr(),
			nodesArr, numSourcesPerRound);

	// Clean up
	cudaFree(nodesArr);
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void Sanity_masterNodes_cuda(struct CUDA_Context* ctx,
		float & DGAccumulator_sum,
		float & DGAccumulator_max,
		float & DGAccumulator_min) {
	// Sizing
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);

	// Accumulators
	HGAccumulator<float> _DGAccumulator_sum;
	HGReduceMax<float> _DGAccumulator_max;
	HGReduceMin<float> _DGAccumulator_min;

	Shared<float> DGAccumulator_sumval  = Shared<float>(1);
	*(DGAccumulator_sumval.cpu_wr_ptr()) = 0;
	_DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr();

	Shared<float> DGAccumulator_maxval  = Shared<float>(1);
	*(DGAccumulator_maxval.cpu_wr_ptr()) = 0;
	_DGAccumulator_max.rv = DGAccumulator_maxval.gpu_wr_ptr();

	Shared<float> DGAccumulator_minval  = Shared<float>(1);
	*(DGAccumulator_minval.cpu_wr_ptr()) = 0;
	_DGAccumulator_min.rv = DGAccumulator_minval.gpu_wr_ptr();

	// Kernel call
	Sanity <<<blocks, threads>>>(
			ctx->gg, ctx->beginMaster, ctx->beginMaster + ctx->numOwned,
			ctx->bc.data.gpu_wr_ptr(),
			_DGAccumulator_sum,
			_DGAccumulator_max,
			_DGAccumulator_min);

	// Clean up
	cudaDeviceSynchronize();
	check_cuda_kernel;

	// Copy back values
	DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr());
	DGAccumulator_max = *(DGAccumulator_maxval.cpu_rd_ptr());
	DGAccumulator_min = *(DGAccumulator_minval.cpu_rd_ptr());
}
