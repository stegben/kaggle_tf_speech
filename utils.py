import os.path as osp

import numpy as np
from scipy.misc import imresize
from scipy.io import wavfile

from constants import TRAIN_AUDIO_PATH
from gen_data import pad_wave_with_zero


MEAN = -17.6725
STD = 3128.12

def read_background(name):
    path = osp.join(TRAIN_AUDIO_PATH, '_background_noise_', name)
    _, wave = wavfile.read(path)
    wave = (wave.astype(np.float32) - MEAN) / STD
    return wave


white_noise = read_background('white_noise.wav')
pink_noise = read_background('pink_noise.wav')
doing_the_dishes = read_background('doing_the_dishes.wav')
dude_miaowing = read_background('dude_miaowing.wav')
exercise_bike = read_background('exercise_bike.wav')
running_tap = read_background('running_tap.wav')


def sample_bg(wave, length, ratio):
    start = np.random.randint(0, (wave.shape[0] - length))
    vol_ratio = np.random.normal(0, ratio)
    return wave[start: start + length] * vol_ratio


def augment(
        arr,
        shift_range=2000,
        speed_ratio=0.2,
        volume_ratio=2,
        white_noise_ratio=0.1,
        pink_noise_ratio=0.1,
        doing_the_dishes_ratio=0.1,
        dude_miaowing_ratio=0.1,
        exercise_bike_ratio=0.1,
        running_tap_ratio=0.1,
    ):
    length = arr.shape[1]
    # shifting
    shifting = np.random.randint(-shift_range, shift_range)
    arr_modified = np.roll(arr, shifting, axis=1)


    for idx in range(arr_modified.shape[0]):
        # speed adjust
        stretched_length = int(length * np.random.uniform(1 - speed_ratio, 1 + speed_ratio))
        arr_modified[idx, :] = pad_wave_with_zero(imresize(
            arr_modified[idx, :],
            size=(1, stretched_length),
        ), length)

        # volume
        # volume_adjust = np.random.uniform(1/volume_ratio, 1*volume_ratio, arr.shape[0])
        volume_adjust = np.random.uniform(1/volume_ratio, 1*volume_ratio)
        arr_modified[idx, :] = arr_modified[idx, :] * volume_adjust

        # with noises
        white_noise_frac = sample_bg(white_noise, length, white_noise_ratio)
        pink_noise_frac = sample_bg(pink_noise, length, pink_noise_ratio)
        doing_the_dishes_frac = sample_bg(doing_the_dishes, length, doing_the_dishes_ratio)
        dude_miaowing_frac = sample_bg(dude_miaowing, length, dude_miaowing_ratio)
        exercise_bike_frac = sample_bg(exercise_bike, length, exercise_bike_ratio)
        running_tap_frac = sample_bg(running_tap, length, running_tap_ratio)

        arr_modified[idx, :] = (
            arr_modified[idx, :]
            + white_noise_frac
            + pink_noise_frac
            + doing_the_dishes_frac
            + dude_miaowing_frac
            + exercise_bike_frac
            + running_tap_frac
        )

    arr_modified = arr_modified
    return arr_modified


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
