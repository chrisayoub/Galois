#ifndef _CUDATREE_
#define _CUDATREE_

#include "slab_hash.cuh"
#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "mrbc_bitset_cuda.cuh"

const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;
const uint32_t num_buckets = 10;
size_t g_gpu_device_idx{0};  // the gpu device to run tests on

using BitSet = CUDABitSet;
class CUDATree : GpuSlabHash<uint32_t, BitSet*, SlabHashTypeT::ConcurrentMap> {
public:
  CUDATree() // double check this. where is this called?
  : GpuSlabHash<uint32_t, BitSet*, SlabHashTypeT::ConcurrentMap>(
          num_buckets, new DynamicAllocatorT(), g_gpu_device_idx) {}
  //! map to a bitset of nodes that belong in a particular distance group


  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! indicates if zero distance has been reached for backward iteration
  bool zeroReached;

	__device__ // __forceinline__?
	void initialize() {
    /* TODO:  slab_hash doesn't support clear
     *        manually delete and create cuda_tree every time? */
    // distanceTree.clear();

    // reset number of sent sources
    numSentSources = 0;
    // reset number of non infinity sources that exist
    numNonInfinity = 0;
    // reset the flag for backward phase
    zeroReached = false;
	}

	__device__
	void setDistance(uint32_t index, uint32_t newDistance, uint32_t tid) {
	  // get distanceTree[newDistance]
    bool toSearch = true;
	  uint32_t laneId = threadIdx.x & 0x1F;
    BitSet *myBitSet = reinterpret_cast<BitSet *>(SEARCH_NOT_FOUND); // is this safe?
    uint32_t myBucket = gpu_context_.computeBucket(newDistance);
    gpu_context_.searchKey(toSearch, laneId, newDistance, myBitSet, myBucket);

    if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
      // newDistance exists
      myBitSet->set_indicator(index);
    } else {
      // newDistance not exists, create myBitSet and write back the pointer
      myBitSet = new BitSet();
      myBitSet->set_indicator(index);

      bool toInsert = true;
      AllocatorContextT local_allocator_ctx(gpu_context_.getAllocatorContext());
      local_allocator_ctx.initAllocator(tid, laneId);
      gpu_context_.insertPair(toInsert, laneId, newDistance, myBitSet, myBucket, local_allocator_ctx);
    }

    numNonInfinity++;
	}

	__device__
	uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    uint32_t indexToSend = infinity;

    bool toSearch = true;
    uint32_t laneId = threadIdx.x & 0x1F;
    BitSet *myBitSet = reinterpret_cast<BitSet *>(SEARCH_NOT_FOUND); // is this safe?
    uint32_t myBucket = gpu_context_.computeBucket(distanceToCheck);
    gpu_context_.searchKey(toSearch, laneId, distanceToCheck, myBitSet, myBucket);

    if (reinterpret_cast<uint64_t>(myBitSet) == SEARCH_NOT_FOUND) {
      auto index = myBitSet->getIndicator();
      if (index != myBitSet->npos) {
        indexToSend = index;
      }
    }

//    CPU code:
//    auto setIter = distanceTree.find(distanceToCheck);
//    if (setIter != distanceTree.end()) {
//      BitSet& setToCheck = setIter->second;
//      auto index = setToCheck.getIndicator();
//      if (index != setToCheck.npos) {
//        indexToSend = index;
//      }
//    }
    return indexToSend;
	}

	__device__
	bool moreWork() { return numNonInfinity > numSentSources; }

	__device__
	void markSent(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    bool toSearch = true;
    uint32_t laneId = threadIdx.x & 0x1F;
    BitSet *myBitSet = reinterpret_cast<BitSet *>(SEARCH_NOT_FOUND);
    uint32_t myBucket = gpu_context_.computeBucket(distanceToCheck);
    gpu_context_.searchKey(toSearch, laneId, distanceToCheck, myBitSet, myBucket);

    myBitSet->forward_indicator();

    numSentSources++;
	}

	__device__
	void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance, uint32_t uid) {
		// TODO
	}

	__device__
	void prepForBackPhase() {
		// TODO
	}

	__device__
	uint32_t backGetIndexToSend(const uint32_t roundNumber,
            const uint32_t lastRound) {
		// TODO
		return 0;
	}

	__device__
	bool isZeroReached() { return zeroReached; }
};

#endif
