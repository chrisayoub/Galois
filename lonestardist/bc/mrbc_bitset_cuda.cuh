
#ifndef _CUDA_BIT_SET_
#define _CUDA_BIT_SET_

#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"
#include "slab_hash.cuh"

class CUDABitSet : DynamicBitset {
  size_t indicator;

  // inline?
  __device__ size_t get_word(size_t pos) const { return pos < 64? 0 : pos / 64; }
  __device__ size_t get_offset(size_t pos) const { return pos < 64? pos : pos % 64; }
  __device__ uint64_t get_mask(size_t pos) const { return uint64_t(1) << get_offset(pos); }

  // adapted from boost::detail::integer_log2_impl
  __device__
  size_t log2(uint64_t x) const {
    int n = 32;
    size_t result = 0;
    while (x != 1) {
      const size_t t = x >> n;
      if (t) {
        result += n;
        x = t;
      }
      n /= 2;
    }
    return result;
  }

  __device__
  size_t right_most_bit(uint64_t w) const {
    // assert(w >= 1);
    return log2(w & -w);
  }

  __device__
  size_t left_most_bit(uint64_t w) const {
    return log2(w);
  }

  __device__
  size_t find_from_block(size_t first, bool fore=true) const {
    size_t i;
    if (fore) {
      // find the position of the first 1-bit starting from block 'first'
      for (i = first; i < vec_size() && bit_vector[i] == 0; i++);
      if (i >= vec_size())
        return npos;
      return i * 64 + right_most_bit(bit_vector[i]);
    }
    else {
      // find the position of the last 1-bit starting from block 'first'
      for (i = first; i > 0 && bit_vector[i] == 0; i--);
      if (i <= 0 && bit_vector[i] == 0)
        return npos;
      return i * 64 + left_most_bit(bit_vector[i]);
    }
  }

  __device__
  size_t find_next(size_t pos) const {
    if (pos == npos) {
      return find_from_block(0); // find_first()
    }
    // ++pos implicitly increment pos
    if (++pos >= size() || size() == 0) {
      return npos;
    }
    size_t curBlock = get_word(pos);
    auto curOffset = get_offset(pos);
    auto seg = bit_vector[curBlock];
    while (seg != bit_vector[curBlock]) // for atomic?
      seg = bit_vector[curBlock];
    uint64_t res = seg >> curOffset;
    return res? pos + right_most_bit(res) : find_from_block(++curBlock);
  }

  __device__
  size_t find_prev(size_t pos) const {
    if (pos >= size()) {
      return find_from_block(vec_size() - 1, false); // find_last()
    }
    // Return npos if no bit set
    if (pos-- == 0 || size() == 0) {
      return npos;
    }
    size_t curBlock = get_word(pos);
    auto curOffset = get_offset(pos);
    auto seg = bit_vector[curBlock];
    uint64_t res = seg & ((uint64_t(2) << curOffset) - 1);
    return res?
           curBlock * 64 + left_most_bit(res) :
           (curBlock? find_from_block(--curBlock, false) : npos);
  }

  __device__
  bool test_set(size_t pos, bool val=true) {
    // TODO: review test_set and the entire CUDA tree
    bool const ret = test(pos);
    if (ret != val) {
      auto word = get_word(pos);
      unsigned long long int mask = get_mask(pos); // why ull?
      uint64_t old_val = bit_vector[word];
      if (val) {
        atomicOr((unsigned long long int*)&bit_vector[word], mask);
      } else {
        atomicAnd((unsigned long long int*)&bit_vector[word], ~mask);
      }
    }
    return ret;
  }

public:
  static const size_t npos = std::numeric_limits<size_t>::max();

  __device__
  CUDABitSet(uint32_t numSources)
  : DynamicBitset((size_t)numSources),
  indicator(npos) {}

  __device__
  size_t getIndicator() {
    return indicator;
  }

  __device__
  void set_indicator(size_t pos) {
    set(pos);
    if (pos < indicator) {
      indicator = pos;
    }
  }

  __device__
  size_t forward_indicator() {
    size_t old = indicator;
    indicator = find_next(indicator);
    return old;
  }

  __device__
  size_t backward_indicator() {
    size_t old = indicator;
    indicator = find_prev(indicator);
    return old;
  }

  __device__
  bool test_set_indicator(size_t pos, bool val=true) {
    if (test_set(pos, val)) {
      if (pos == indicator) {
        forward_indicator();
      }
      return true;
    }
    return false;
  }

  __device__
  bool none() {
    for (size_t i = 0; i < vec_size(); ++i)
      if (bit_vector[i])
        return false;
    return true;
  }

  __device__
  bool nposInd() {
    return indicator == npos;
  }

};

#endif