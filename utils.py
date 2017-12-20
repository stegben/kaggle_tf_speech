import numpy as np


class BatchGenerator(object):

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 32,
            shuffle: bool = True,
            seed: int = 2017,
        ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.data_size = x.shape[0]
        self.shuffle = shuffle
        if shuffle:
            np.random.seed(seed)

    def __call__(self):
        if self.shuffle:
            data_index = np.random.permutation(self.data_size)
        else:
            data_index = np.arange(self.data_size)

        for start_idx in range(0, self.data_size, self.batch_size):
            end_idx = start_idx + self.batch_size
            target_idxs = data_index[start_idx: end_idx]
            yield self.x[target_idxs], self.y[target_idxs]
