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
class CUDATree : public GpuSlabHash<uint32_t, BitSet*, SlabHashTypeT::ConcurrentMap> {
  __device__
  BitSet * get(uint32_t myDistance, uint32_t tid) {
    BitSet *myBitSet = search(myDistance);
    if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
      myBitSet = new BitSet();
      insert(myDistance, myBitSet, tid);
    }
    return myBitSet;
  }

  __device__
  BitSet * search(uint32_t myDistance) {
    bool toSearch = true;
    uint32_t laneId = threadIdx.x & 0x1F;
    BitSet *myBitSet = reinterpret_cast<BitSet *>(SEARCH_NOT_FOUND);
    uint32_t myBucket = gpu_context_.computeBucket(myDistance);
    gpu_context_.searchKey(toSearch, laneId, myDistance, myBitSet, myBucket);
    return myBitSet;
  }

  __device__
  void insert(uint32_t myDistance, BitSet* myBitSet, uint32_t tid) {
    bool toInsert = true;
    uint32_t laneId = threadIdx.x & 0x1F;
    uint32_t myBucket = gpu_context_.computeBucket(myDistance);
    AllocatorContextT local_allocator_ctx(gpu_context_.getAllocatorContext());
    local_allocator_ctx.initAllocator(tid, laneId);
    gpu_context_.insertPair(toInsert, laneId, myDistance, myBitSet, myBucket, local_allocator_ctx);
  }

  __device__
  bool notExistOrEmpty(uint32_t myDistance) {
    BitSet *myBitSet = search(myDistance);
    // since we iterate distances by value not iterator,
    // some distances may not exist in the hash map
    if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
      return true;
    }
    return myBitSet->none();
  }

  __device__
  void clear() {
    uint32_t laneId = threadIdx.x & 0x1F;
    for (int i = maxDistance; i >= 0; i--) {
      BitSet *myBitSet = search(i);
      if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
        bool toDelete = true;
        uint32_t myBucket = gpu_context_.computeBucket(i);
        gpu_context_.deleteKey(toDelete, laneId, i, myBucket);
        delete myBitSet;
      }
    }
  }

  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! indicates if zero distance has been reached for backward iteration
  bool zeroReached;

  // distanceTree.rbegin(), curKey, endKey
  uint32_t maxDistance, curDistance, endDistance;

  // need to pass from host somewhere!
  uint32_t numSources;

public:
  CUDATree() // where is CUDATree instantiated?
          : GpuSlabHash<uint32_t, BitSet*, SlabHashTypeT::ConcurrentMap>(
          num_buckets, new DynamicAllocatorT(), g_gpu_device_idx) {}
  //! map to a bitset of nodes that belong in a particular distance group

	__device__ // __forceinline__?
	void initialize() {
    clear();

    // reset number of sent sources
    numSentSources = 0;
    // reset number of non infinity sources that exist
    numNonInfinity = 0;
    // reset the flag for backward phase
    zeroReached = false;

    maxDistance = 0;
	}

	__device__
	void setDistance(uint32_t index, uint32_t newDistance, uint32_t tid) {
    // Only for iterstion initialization
    // assert(newDistance == 0);
    // assert(distanceTree[newDistance].size() == numSourcesPerRound);

    get(newDistance, tid)->set_indicator(index);
    // maxDistance = maxDistance > newDistance ? maxDistance : newDistance;

    numNonInfinity++;
	}

	__device__
	uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    uint32_t indexToSend = infinity;

    BitSet *myBitSet = search(distanceToCheck);
    if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
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
	void markSent(uint32_t roundNumber, uint32_t tid) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    get(distanceToCheck, tid)->forward_indicator();
    numSentSources++;
	}

	__device__
	void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance, uint32_t tid) {
	  // taken care of by callee, oldDistance always > newDistance
//    if (oldDistance == newDistance) {
//      return;
//    }
    maxDistance = maxDistance > newDistance ? maxDistance : newDistance;

    BitSet *myBitSet = search(oldDistance);
    bool existed = false;
    // if it exists, remove it // shouldn't it always exist?
    if (reinterpret_cast<uint64_t>(myBitSet) != SEARCH_NOT_FOUND) {
      existed = myBitSet->test_set_indicator(index, false); // Test, set, update
    }

    // if it didn't exist before, add to number of non-infinity nodes
    if (!existed) {
      numNonInfinity++;
    }

    // asset(distanceTree[newDistance].size() == numSourcesPerRound);
    get(newDistance, tid)->set_indicator(index);
	}

	__device__
	void prepForBackPhase() {
    curDistance = maxDistance;
    endDistance = -1; // or 0? need to worry about overflow?

    if (curDistance != endDistance) {
      // find non-empty distance if first one happens to be empty
      if (notExistOrEmpty(curDistance)) {
        for (--curDistance; curDistance != endDistance && notExistOrEmpty(curDistance); --curDistance);
      }
    }

    // setup if not empty
    if (curDistance != endDistance) {
      search(curDistance)->backward_indicator();
    }
	}

	__device__
	uint32_t backGetIndexToSend(const uint32_t roundNumber,
                              const uint32_t lastRound) {
    uint32_t indexToReturn = infinity;

    while (curDistance != endDistance) {
      if ((curDistance + numSentSources - 1) != (lastRound - roundNumber)){
        // round to send not reached yet; get out
        return infinity;
      }

      if (curDistance == 0) {
        zeroReached = true;
        return infinity;
      }

      BitSet* curSet = search(curDistance);
      if (!curSet->nposInd()) {
        // this number should be sent out this round
        indexToReturn = curSet->backward_indicator();
        numSentSources--;
        break;
      } else {
        // set exhausted; go onto next set
        for (--curDistance; curDistance != endDistance && notExistOrEmpty(curDistance); --curDistance);

        // if another set exists, set it up, else do nothing
        if (curDistance != endDistance) {
          search(curDistance)->backward_indicator();
        }
      }
    }

    if (curDistance == endDistance) {
      assert(numSentSources == 0);
    }
		return indexToReturn;
	}

	__device__
	bool isZeroReached() { return zeroReached; }
};

#endif
