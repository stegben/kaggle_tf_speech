import pickle as pkl

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from model import WaveletNeuralNetworkClassifier
from utils import BatchGenerator
from constants import (
    GENERATED_RAW_DATA_PATH,
    MODEL_DIR,
)


def main():
    # with open(GENERATED_RAW_DATA_PATH, 'rb') as f:
    with open('../tf_speech_generated_data/RAW_DATA_20171220-083411_.pkl', 'rb') as f:
        data = pkl.load(f)

    train, val, test, _ = data
    x_train, label_train = train
    x_val, label_val = val
    x_test, label_test = test

    x_train = x_train.astype(np.float32)
    mean = x_train.mean()
    std = x_train.std()
    print(mean)
    print(std)
    x_train = (x_train - mean) / std
    x_val = x_val.astype(np.float32)
    x_val = (x_val - mean) / std
    x_test = x_test.astype(np.float32)
    x_test = (x_test - mean) / std

    lb = LabelBinarizer()
    y_train = lb.fit_transform(label_train).astype(np.float32)
    y_val = lb.transform(label_val).astype(np.float32)
    y_test = lb.transform(label_test).astype(np.float32)
    # import ipdb; ipdb.set_trace()
    batch_gen = BatchGenerator(x_train, y_train, batch_size=32)
    clf = WaveletNeuralNetworkClassifier(
        x_train.shape[1],
        n_wavelets=32,
        wavelet_range=[k for k in range(1, 17)],
        wavelet_length=16,
        conv_structure=[
            (2, 8, 1, 2, 32, 'selu'),
            (2, 8, 1, 2, 32, 'selu'),
            (1, 4, 1, 4, -1, 'pooling'),
            (2, 8, 1, 4, 64, 'selu'),
            (2, 8, 1, 4, 64, 'selu'),
            (2, 4, 2, 4, -1, 'pooling'),
        ],
        dense_structure=[
            (512, 'selu'),
        ],
        output_dim=y_train.shape[1],
        wavelet_dropout_prob=0.0,
        conv_dropout_prob=0.0,
        dense_dropout_prob=0.5,
    )
    clf.fit_generator(
        batch_gen,
        x_subtrain=x_train,
        y_subtrain=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=100,
        learning_rate=0.0003,
        early_stopping_rounds=10,
        save_best=True,
        save_folder=MODEL_DIR,
    )
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
