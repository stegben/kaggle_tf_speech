import os
import pickle as pkl
import glob
import random

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from constants import (
    TRAIN_AUDIO_PATH_PATTERN,
    TEST_AUDIO_PATH_PATTERN,
    TRAIN_VAL_PATH,
    TRAIN_TEST_PATH,
    N_SAMPLE,
    SILENCE_SLIDING_LENGTH,
    TARGET_LABELS,
    UNKNOWN_LABEL,
    SILENCE_LABEL,
    GENERATED_RAW_DATA_PATH,
)


def is_silence(path):
    if path == '_background_noise_':
        return True
    return False


def gen_label(label_name):
    if label_name in TARGET_LABELS:
        return label_name
    else:
        return UNKNOWN_LABEL


def pad_wave_with_zero(wave, length):
    wave_length = wave.shape[0]
    if wave_length < length:
        return np.lib.pad(
            wave,
            (length-wave_length, 0),
            mode='constant',
            constant_values=0,
        )
    return wave[:length]


def read_target_wave(path, length=None):
    _, wave = wavfile.read(path)

    wave_dim = wave.shape

    if len(wave_dim) != 1:
        raise ValueError('wave can only be one dimention, but {} has dimension: {}'.format(
            path,
            wave_dim,
        ))

    if length is None:
        return wave

    if wave_dim[0] != length:
        print("{} does not have {} points, its shape is: {}".format(
            path,
            length,
            wave_dim,
        ))
        return pad_wave_with_zero(wave, length)
    return wave


def read_silence_wave(path, length, window):
    wave = read_target_wave(path)
    max_length = wave.shape[0]
    for start_idx in range(0, max_length, window):
        end_idx = start_idx + length
        yield pad_wave_with_zero(wave[start_idx: end_idx], length=length)


def main():
    x_train = []
    label_train = []
    x_val = []
    label_val = []
    x_test = []
    label_test = []

    print('training data')
    for path in glob.glob(TRAIN_AUDIO_PATH_PATTERN):
        print(path)
        # file_name = path.split('/')[-1]
        label_name = path.split('/')[-2]
        if is_silence(label_name):
            print('process silence file: {}'.format(path))
            for wave in read_silence_wave(path, N_SAMPLE, SILENCE_SLIDING_LENGTH):
                dice = random.random()
                if dice < 0.79:
                    x_train.append(wave)
                    label_train.append(SILENCE_LABEL)
                elif 0.79 <= dice < 0.895:
                    x_val.append(wave)
                    label_val.append(SILENCE_LABEL)
                else:
                    x_test.append(wave)
                    label_test.append(SILENCE_LABEL)
        else:
            wave = read_target_wave(path, N_SAMPLE)
            if path in TRAIN_VAL_PATH:
                x_val.append(wave)
                label_val.append(gen_label(label_name))
            elif path in TRAIN_TEST_PATH:
                x_test.append(wave)
                label_test.append(gen_label(label_name))
            else:
                x_train.append(wave)
                label_train.append(gen_label(label_name))

    x_train = np.vstack(x_train)
    x_val = np.vstack(x_val)
    x_test = np.vstack(x_test)

    print('submission data')
    x_sub = []
    fname_sub = []
    for path in tqdm(glob.glob(TEST_AUDIO_PATH_PATTERN)):
        wave = read_target_wave(path, N_SAMPLE)
        fname = path.split('/')[-1]
        x_sub.append(wave)
        fname_sub.append(fname)
    x_sub = np.vstack(x_test)

    with open(GENERATED_RAW_DATA_PATH, 'wb') as f:
        pkl.dump((
            (x_train, label_train),
            (x_val, label_val),
            (x_test, label_test),
            (x_sub, fname_sub),
        ), f)


if __name__ == '__main__':
    main()


