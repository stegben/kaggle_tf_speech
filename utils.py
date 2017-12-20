import os.path as osp

import numpy as np
from scipy.io import wavfile

from constants import TRAIN_AUDIO_PATH


def read_background(name):
    path = osp.join(TRAIN_AUDIO_PATH, '_background_noise_', name)
    wave = read_target_wave(path).astype(np.float32) /
    return wave


white_noise = read_background('white_noise.wav')
pink_noise = read_background('pink_noise.wav')
doing_the_dishes = read_background('doing_the_dishes.wav')
dude_miaowing = read_background('dude_miaowing.wav')
exercise_bike = read_background('exercise_bike.wav')
running_tap = read_background('running_tap.wav')


def augment(
        arr,
        shift_range=2000,
        speed_ratio=0.2,
        volume_ratio=3,
        white_noise=,
        pink_noise,
        doing_the_dishes,
        dude_miaowing,
        exercise_bike,
        running_tap,
    ):
    shifting =
    volume_adjust = np.random.uniform(1/volume_ratio, 1*volume_ratio, arr.shape[0])
    return arr *

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
            yield (
                augment(self.x[target_idxs]),
                self.y[target_idxs],
            )
