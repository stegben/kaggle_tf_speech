import pickle as pkl

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from model import WaveletNeuralNetworkClassifier
from utils import BatchGenerator
from constants import (
    GENERATED_RAW_DATA_PATH,
)


def main():
    # with open(GENERATED_RAW_DATA_PATH, 'rb') as f:
    with open('../tf_speech_generated_data/RAW_DATA_20171219-225647_.pkl', 'rb') as f:
        data = pkl.load(f)

    train, val, test, _ = data
    x_train, label_train = train
    x_val, label_val = val
    x_test, label_test = test

    x_train = x_train.astype(np.float32)
    mean = x_train.mean()
    std = x_train.std()
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
        output_dim=y_train.shape[1],
    )
    clf.fit_generator(
        batch_gen,
        x_val=x_val,
        y_val=y_val,
        epochs=100,
        learning_rate=0.0003,
        early_stopping_rounds=10,
        save_best=True,
    )
    # clf = WaveletNeuralNetworkClassifier(
    #     16000,
    #     n_wavelets=32,
    #     output_dim=12,
    # )
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
