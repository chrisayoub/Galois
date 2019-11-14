
const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

class CUDATree {
	// TODO implement this to be MRBCTree but for CUDA
public:
	__device__ void initialize() {
		// TODO
	}

	__device__  void setDistance(uint32_t index, uint32_t newDistance) {
		// TODO
	}
};
