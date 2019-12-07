#include "bc_mr_cuda.cuh"
#include <new>

// *******************************
// ** Helper functions (device code)
// ********************************

__device__
unsigned flatMapArraySize;

// 2D arrays are being represented as flat-maps. This gives us the index for sourceData array
__device__
unsigned getArrayIndex(unsigned node, unsigned index) {
	return node * flatMapArraySize + index;
}

// *************************
// ** Kernels (device code)
// *************************

// Needed to set device field
__global__
void SetVectorSize(unsigned vectorSize) {
	flatMapArraySize = vectorSize;
}

__global__
void InitializeIteration(
		CSRGraph graph,
		unsigned int __begin, unsigned int __end,
		uint32_t* p_roundIndexToSend,
		CUDATree* p_dTree,
		uint32_t* p_minDistances,
		ShortPathType* p_shortPathCounts,
		float* p_dependencyValues,
		uint64_t* nodesToConsider, unsigned numSourcesPerRound)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  for (index_type src = __begin + tid; src < __end; src += nthreads)
  {
	  p_roundIndexToSend[src] = infinity;
	  CUDATree& dTree = p_dTree[src]; // CUDATree -> CUDATree& to avoid value copy
	  dTree.initialize(numSourcesPerRound);

	  // Loop through sources
	  for (unsigned i = 0; i < numSourcesPerRound; i++) {
		  unsigned idx = getArrayIndex(src, i);
		  if (nodesToConsider[i] == graph.node_data[src]) {
			  // This is a source node
			  p_minDistances[idx] = 0;
			  p_shortPathCounts[idx] = 1;
			  p_dependencyValues[idx] = 0.0;
			  dTree.setDistance(i, 0);
		  } else {
			  // This is a non-source node
			  p_minDistances[idx] = infinity;
			  p_shortPathCounts[idx] = 0;
			  p_dependencyValues[idx] = 0.0;
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
		uint32_t* p_minDistances,
		DynamicBitset& bitset_minDistances) {

	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	__shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
	dga.thread_entry();

	for (index_type src = __begin + tid; src < __end; src += nthreads)
	{
		uint32_t* roundIndexToSend = &p_roundIndexToSend[src];
		CUDATree& dTree = p_dTree[src];

		uint32_t newRoundIndex = dTree.getIndexToSend(roundNumber);
		*roundIndexToSend = newRoundIndex;

		if (newRoundIndex != infinity) {
			unsigned srcIndex = getArrayIndex(src, newRoundIndex);
			if (p_minDistances[srcIndex] != 0) {
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
		  CUDATree& dTree = p_dTree[src];
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
		uint32_t* p_minDistances,
		ShortPathType* p_shortPathCounts)
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
			index_type src = graph.getAbsDestination(edge);
			uint32_t indexToSend = p_roundIndexToSend[src];

			unsigned destIndex = getArrayIndex(dest, indexToSend);
			unsigned srcIndex = getArrayIndex(src, indexToSend);

			if (indexToSend != infinity) {
				uint32_t distValue = p_minDistances[srcIndex];
				uint32_t newValue = distValue + 1;
			    // Update minDistance vector
				uint32_t oldValue = p_minDistances[destIndex];

				if (oldValue > newValue) {
					p_minDistances[destIndex] = newValue;
					p_dTree[dest].setDistance(indexToSend, oldValue, newValue);
					// overwrite short path with this node's shortest path
					p_shortPathCounts[destIndex] = p_shortPathCounts[srcIndex];
				} else if (oldValue == newValue) {
					// add to short path
					p_shortPathCounts[destIndex] += p_shortPathCounts[srcIndex];
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
		float* p_dependencyValues,
		DynamicBitset& bitset_dependency)
{
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type src = __begin + tid; src < __end; src += nthreads) {
		// if zero distances already reached, there is no point sending things
		// out since we don't care about dependecy for sources (i.e. distance
		// 0)
		CUDATree& dTree = p_dTree[src];
		if (!dTree.isZeroReached()) {
			uint32_t newRoundIndex = dTree.backGetIndexToSend(roundNumber, lastRoundNumber);
			p_roundIndexToSend[src] = newRoundIndex;

			if (newRoundIndex != infinity) {
	            // only comm if not redundant 0
				unsigned srcIndex = getArrayIndex(src, newRoundIndex);
				if (p_dependencyValues[srcIndex] != 0) {
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
		uint32_t* p_minDistances,
		ShortPathType* p_shortPathCounts,
		float* p_dependencyValues) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type dest = __begin + tid; dest < __end; dest += nthreads) {
		unsigned i = p_roundIndexToSend[dest];

		if (i != infinity) {
			unsigned destIndex = getArrayIndex(dest, i);

			uint32_t myDistance = p_minDistances[destIndex];

		    // calculate final dependency value
			p_dependencyValues[destIndex] = p_dependencyValues[destIndex] * p_shortPathCounts[destIndex];

			// get the value to add to predecessors
			float toAdd = ((float)1 + p_dependencyValues[destIndex]) /
					p_shortPathCounts[destIndex];

			// Loop through current node's edges
			index_type edge_start = graph.getFirstEdge(dest);
			index_type edge_end = graph.getFirstEdge(dest + 1);
			for (index_type edge = edge_start; edge < edge_end; edge++)
			{
				index_type src = graph.getAbsDestination(edge);
				unsigned srcIndex = getArrayIndex(src, i);
				uint32_t sourceDistance = p_minDistances[srcIndex];

				// source nodes of this batch (i.e. distance 0) can be safely
				// ignored
				if (sourceDistance != 0) {
					// determine if this source is a predecessor
					if (myDistance == (sourceDistance + 1)) {
						// add to dependency of predecessor using our finalized one
						atomicTestAdd(&p_dependencyValues[srcIndex], toAdd);
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
		float* p_dependencyValues,
		uint64_t* nodesToConsider, unsigned numSourcesPerRound) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;

	for (index_type src = __begin + tid; src < __end; src += nthreads) {
		for (unsigned i = 0; i < numSourcesPerRound; i++) {
			// exclude sources themselves from BC calculation
			if (graph.node_data[src] != nodesToConsider[i]) {
				unsigned srcIndex = getArrayIndex(src, i);
				p_bc[src] += p_dependencyValues[srcIndex];
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
	check_cuda_kernel;
	return arr;
}

// Need to reduce number of threads so it works
void size_kernel(dim3& blocks, dim3& threads) {
	kernel_sizing(blocks, threads);
	threads.x = 1;
	blocks.x = 1;
}

// *******************************
// ** Kernel wrappers (host code)
// ********************************

void FinishMemoryInit_cuda(struct CUDA_Context* ctx, unsigned vectorSize) {
	// Init the fields
	ctx->vectorSize = vectorSize;

	// Custom so we can make flat-map arrays
	unsigned num_hosts = ctx->num_hosts;
	load_array_field_CUDA(ctx, &ctx->minDistances, num_hosts);
	load_array_field_CUDA(ctx, &ctx->shortPathCounts, num_hosts);
	load_array_field_CUDA(ctx, &ctx->dependencyValues, num_hosts);

	// Copy vectorSize to device for utility
//	SetVectorSize<<<1, 1>>>(vectorSize);

	// Ensure we have enough heap space for malloc in device code
	// For each node, we have a CUDAMap (stored in dTree)
	// For each CUDAMap, we have up to 2 * sizeof(MapPair) * N entries.
	unsigned N = ctx->gg.nnodes;
	size_t heapSpace = 2 * sizeof(MapPair) * N * N;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSpace);

	// Clear the dTree storage (only doing this once)
	ctx->dTree.data.zero_gpu();

	// Finish op
	cudaDeviceSynchronize();
	check_cuda_kernel;
}

void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx)
{
	// Init memory
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
	size_kernel(blocks, threads);
	printf("Blocks %d %d %d \n", blocks.x, blocks.y, blocks.z);
	printf("Threads %d %d %d \n", threads.x, threads.y, threads.z);


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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
	size_kernel(blocks, threads);

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
