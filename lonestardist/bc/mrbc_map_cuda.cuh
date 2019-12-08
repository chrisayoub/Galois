#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "mrbc_bitset_cuda.cuh"
#include <new>

using BitSet = CUDABitSet;

// Used for storage
struct MapPair {
	uint32_t key;
	BitSet* value;
	uint8_t used; // Indicates if this spot is used
};
using MapPair = struct MapPair;

class CUDAMap {

private:

	// Internal storage, array of MapPair
	MapPair* map;
	// Number of elements
	unsigned size;
	// Overall capacity
	unsigned length;
	// Needed for bitsets
	uint32_t numSources;

	// Resize exponentially
	__device__
	void resize() {
		// Currently fails on resizing, need more work for this...
		// Can't free() on device if cudaMalloc() on host
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-allocation-and-lifetime
		assert(size < length - 1);
	}

	// Return the hash of a key
	// Will need to mod the result by whatever array length
	__device__
	uint32_t hash(uint32_t key) {
		// TODO make better hash function, maybe?
		return key;
	}

	// Insert a key-value into the given storage
	// Pre-condition: enough space exists in the storage
	__device__
	void insert(uint32_t key, BitSet* value, MapPair* storage, unsigned storageLength) {
		// Get the spot
		unsigned index = findSpot(key, storage, storageLength);
		// Now insert
		storage[index].key = key;
		storage[index].value = value;
		storage[index].used = 1;
	}

	// Find the spot that this key should be in, or is in if present
	__device__
	unsigned findSpot(uint32_t key, MapPair* storage, unsigned storageLength) {
		unsigned index = hash(key) % storageLength;

		// while element exists...
		while (storage[index].used != 0) {
			if (storage[index].key == key) {
				// if this is the element, return it
				return index;
			}
			// otherwise go to next place
			index = (index + 1) % storageLength;
		}
		// empty spot for element
		return index;
	}

public:
	__host__
	static CUDAMap* getDeviceMap(uint32_t numSources, unsigned capacity) {
		// Allocate a map on the device
		CUDAMap* deviceMap;
		cudaMalloc(&deviceMap, sizeof(CUDAMap));
		check_cuda_kernel;

		// Initial values of map
		cudaMemcpy(&deviceMap->length, &capacity, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(&deviceMap->numSources, &numSources, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemset(&deviceMap->size, 0, sizeof(unsigned));

		// Allocate actual array
		MapPair* deviceArray;
		size_t numBytes = sizeof(MapPair) * capacity;
		cudaMalloc(&deviceArray, numBytes);
		check_cuda_kernel;
		cudaMemset(deviceArray, 0, numBytes);

		// Set pointer to array
		cudaMemcpy(&deviceMap->map, &deviceArray, sizeof(MapPair*), cudaMemcpyHostToDevice);

		// Finish ops
		cudaDeviceSynchronize();

		// Return device pointer
		return deviceMap;
	}

	// Return nullptr if not present
	__device__
	BitSet* search(uint32_t key) {
		unsigned index = findSpot(key, map, length);
		if (map[index].used != 0) {
			// Present!
			return map[index].value;
		}
		return nullptr;
	}

	// Insert new kv into the map
	__device__
	void insert(uint32_t key, BitSet* value) {
		resize();
		insert(key, value, map, length);
		size++;
	}

	// Look for existing bitset. If not found, create new one and return it
	__device__
	BitSet* get(uint32_t key) {
		resize();
		unsigned index = findSpot(key, map, length);
		if (map[index].used == 0) {
			// Need to create new bitset and place
			map[index].key = key;
			map[index].value = new CUDABitSet(numSources);
			map[index].used = 1;
			size++;
		}
		return map[index].value;
	}

	// Clear all the positions, free all the bitsets
	__device__
	void clear() {
		for (unsigned i = 0; i < length; i++) {
			if (map[i].used) {
				// Free memory
				BitSet* val = map[i].value;
				delete val;

				map[i].key = 0;
				map[i].value = 0;
				map[i].used = 0;
			}
		}
		size = 0;
	}

	// Return true if not exists, or if empty value
	__device__
	bool notExistOrEmpty(uint32_t key) {
		BitSet* result = search(key);
		if (result == nullptr) {
			return true;
		}
		return result->none();
	}
};

