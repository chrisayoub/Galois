
const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

class CUDATree {
	// TODO implement this to be MRBCTree but for CUDA
public:
	__device__
	void initialize() {
		// TODO
	}

	__device__
	void setDistance(uint32_t index, uint32_t newDistance) {
		// TODO
	}

	__device__
	uint32_t getIndexToSend(uint32_t roundNumber) {
		// TODO
	}

	__device__
	bool moreWork() {
		// TODO
	}

	__device__
	void markSent(uint32_t roundNumber) {
		// TODO
	}

	__device__
	void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
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
	}

	__device__
	bool isZeroReached() {
		// TODO
	}
};
