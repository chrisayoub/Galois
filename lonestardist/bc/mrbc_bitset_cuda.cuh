
#ifndef _CUDA_BIT_SET_
#define _CUDA_BIT_SET_

#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "slab_hash.cuh"

class CUDABitSet {
  size_t indicator;
public:
  bool notFound;
  static const size_t npos = std::numeric_limits<size_t>::max();

  __device__
  CUDABitSet(uint32_t size = 0) {
    if (size == SEARCH_NOT_FOUND) notFound = true;
  }

  __device__
  size_t getIndicator() {
    return indicator;
  }

  __device__
  void set_indicator(size_t pos) {

  }

  __device__
  void forward_indicator() {

  }

};

#endif