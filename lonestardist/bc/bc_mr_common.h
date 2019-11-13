#include "galois/AtomicWrapper.h"

// type of short path
using ShortPathType = double;

/**
 * Structure for holding data calculated during BC
 */
struct BCData {
  uint32_t minDistance;
  ShortPathType shortPathCount;
  galois::CopyableAtomic<float> dependencyValue;
};
