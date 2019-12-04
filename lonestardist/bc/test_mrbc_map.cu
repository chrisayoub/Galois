#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>

#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "galois/cuda/DynamicBitset.h"
#include "mrbc_map_cuda.cuh"

#define DEVICE_ID 0

__global__
void testMap() {
	uint32_t NUM_SRC = 10;
	CUDAMap* map = new CUDAMap(NUM_SRC);

	uint32_t KEY = 3;
	if (map->get(KEY) != nullptr) {
		printf("ERROR: should not contain key yet \n");
		return;
	}

	BitSet* b = map->get(3);
	if (map->get(KEY) == nullptr) {
		printf("ERROR: should contain key \n");
		return;
	}

	if (!map->notExistOrEmpty(KEY)) {
		printf("ERROR: is actually empty \n");
		return;
	}

	map->get(KEY)->set_indicator(1);
	if (map->notExistOrEmpty(KEY)) {
		printf("ERROR: is NOT empty \n");
		return;
	}

	map->clear();
	if (map->get(KEY) != nullptr) {
		printf("ERROR: should not contain key after clear \n");
		return;
	}

	// Now try resizing with many inserts, ensure no crach
	for (uint32_t i = 0; i < 50; i++) {
		map->get(i);
	}

	if (map->get(KEY + 5) == nullptr) {
		printf("ERROR: should contain key \n");
		return;
	}

	printf("All map cases passed!");
}

int main(int argc, char** argv) {
  //=========
  int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if (devCount) {
    cudaSetDevice(DEVICE_ID);  // be changed later
    cudaGetDeviceProperties(&devProp, DEVICE_ID);
  }
  printf("Device: %s\n", devProp.name);

  // Call kernel that will use map
  testMap<<<1, 1>>>();

  return 0;
}
