from arch.bootstrap import StationaryBootstrap, IIDBootstrap, CircularBlockBootstrap, MovingBlockBootstrap, IndependentSamplesBootstrap
import numpy as np

try:
    from arch.bootstrap._samplers import stationary_bootstrap_sample
except ImportError:  # pragma: no cover
    from arch.bootstrap._samplers_python import stationary_bootstrap_sample


# Переопределены методы генерации семплов в StationaryBootstrap,
# CircularBlockBootstrap, MovingBlockBootstrap

class CircularBlockBootstrapM(CircularBlockBootstrap):
    def __init__(self, samples, block_size, *args, **kwargs):
        super().__init__(block_size, *args, **kwargs)
        self.set_samples(samples)

    def set_samples(self, samples):
        self._samples = samples    

    def update_indices(self):
        num_blocks = self._samples // self.block_size
        if num_blocks * self.block_size < self._samples:
            num_blocks += 1

        indices = self.random_state.randint(self._num_items, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()
        indices %= self._num_items

        if indices.shape[0] > self._samples:
            return indices[: self._samples]
        else:
            return indices

# class StationaryBootstrapM(StationaryBootstrap):
#     def __init__(self, samples, block_size, *args, **kwargs):
#         super().__init__(block_size, *args, **kwargs)
#         self.set_samples(samples)

#     def set_samples(self, samples):
#         self._samples = samples    

#     def update_indices(self):
#         indices = self.random_state.randint(self._num_items, size=self._samples)
#         indices = indices.astype(np.int64)
#         u = self.random_state.random_sample(self._num_items)

#         indices = stationary_bootstrap_sample(indices, u, self._p)
#         return indices


class MovingBlockBootstrapM(MovingBlockBootstrap):
    def __init__(self, samples, block_size, *args, **kwargs):
        super().__init__(block_size, *args, **kwargs)
        self.set_samples(samples)

    def set_samples(self, samples):
        self._samples = samples    

    def update_indices(self):
        num_blocks = self._samples // self.block_size
        if num_blocks * self.block_size < self._samples:
            num_blocks += 1

        max_index = self._num_items - self.block_size + 1
        indices = self.random_state.randint(max_index, size=num_blocks)
        indices = indices[:, None] + np.arange(self.block_size)
        indices = indices.flatten()

        if indices.shape[0] > self._samples:
            return indices[: self._samples]
        else:
            return indices

