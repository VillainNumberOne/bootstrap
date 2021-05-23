from arch.bootstrap import StationaryBootstrap, IIDBootstrap, CircularBlockBootstrap, MovingBlockBootstrap, IndependentSamplesBootstrap
import numpy as np

class CircularBlockBootstrapM(CircularBlockBootstrap):
    def __init__(self, samples, block_size, *args, **kwargs):
        super().__init__(block_size, *args, **kwargs)
        self.set_samples(samples)

    def set_samples(self, samples):
        assert samples <= self._num_items, "didn't fixed yet"
        self._samples = samples    

    def update_indices(self):
        num_blocks = self._samples // self.block_size
        if num_blocks * self.block_size < self._samples:
            num_blocks += 1
        indices = self.random_state.randint(self._samples, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()
        indices %= self._samples

        if indices.shape[0] > self._samples:
            return indices[: self._samples]
        else:
            return indices