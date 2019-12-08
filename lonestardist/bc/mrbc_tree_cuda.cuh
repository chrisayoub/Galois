#ifndef _CUDATREE_
#define _CUDATREE_

#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "mrbc_bitset_cuda.cuh"
#include "mrbc_map_cuda.cuh"

const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

using BitSet = CUDABitSet;
class CUDATree {
	__device__
	BitSet* get(uint32_t myDistance) {
		return map->get(myDistance);
	}

	__device__
	BitSet* search(uint32_t myDistance) {
		return map->search(myDistance);
	}

	__device__
	bool notExistOrEmpty(uint32_t myDistance) {
		return map->notExistOrEmpty(myDistance);
	}

	__device__
	void clear() {
		map->clear();
	}

	// internal map used for hashing/storage
	CUDAMap* map;

	//! number of sources that have already been sent out
	uint32_t numSentSources;
	//! number of non-infinity values (i.e. number of sources added already)
	uint32_t numNonInfinity;
	//! indicates if zero distance has been reached for backward iteration
	bool zeroReached;

	// distanceTree.rbegin(), curKey, endKey
	uint32_t maxDistance, curDistance, endDistance;

public:
	__host__
	void setMap(CUDAMap* deviceMap) {
		// Set pointer on device from tree to map
		cudaMemcpy(&map, &deviceMap, sizeof(CUDAMap*), cudaMemcpyHostToDevice);
	}

	__device__
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
	void setDistance(uint32_t index, uint32_t newDistance) {
		get(newDistance)->set_indicator(index);
		numNonInfinity++;
	}

	__device__
	uint32_t getIndexToSend(uint32_t roundNumber) {
		uint32_t distanceToCheck = roundNumber - numSentSources;
		uint32_t indexToSend = infinity;

		BitSet *setToCheck = search(distanceToCheck);
		if (setToCheck != nullptr) {
			auto index = setToCheck->getIndicator();
			if (index != setToCheck->npos) {
				indexToSend = index;
			}
		}

		return indexToSend;
	}

	__device__
	bool moreWork() { return numNonInfinity > numSentSources; }

	__device__
	void markSent(uint32_t roundNumber) {
		uint32_t distanceToCheck = roundNumber - numSentSources;
		get(distanceToCheck)->forward_indicator();
		numSentSources++;
	}

	__device__
	void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
		// taken care of by callee, oldDistance always > newDistance
		maxDistance = maxDistance > newDistance ? maxDistance : newDistance;

		BitSet *setToChange = search(oldDistance);
		bool existed = false;
		// if it exists, remove it // shouldn't it always exist?
		if (setToChange != nullptr) {
			existed = setToChange->test_set_indicator(index, false); // Test, set, update
		}

		// if it didn't exist before, add to number of non-infinity nodes
		if (!existed) {
			numNonInfinity++;
		}

		get(newDistance)->set_indicator(index);
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
