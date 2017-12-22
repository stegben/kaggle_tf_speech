from collections import Counter
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

with open('../tf_speech_generated_data/RAW_DATA_20171220-083411_.pkl', 'rb') as f:
    data = pkl.load(f)

train, val, test, _ = data
x_train, label_train = train
x_val, label_val = val
x_test, label_test = test

x_train = x_train.astype(np.float32)
mean = x_train.mean()
std = x_train.std()
# x_train = (x_train - mean) / std
x_train = x_train / std
# x_train = ((x_train.T - x_train.mean(axis=1)) / x_train.std(axis=1)).T

label_count = Counter(label_train)
sample_weight_train = np.array([2000/label_count[label] for label in label_train], dtype=np.float32)

x_val = x_val.astype(np.float32)
x_val = x_val / std
# x_val = (x_val - mean) / std
x_test = x_test.astype(np.float32)
x_test = x_test / std
# x_test = (x_test - mean) / std

lb = LabelBinarizer()
y_train = lb.fit_transform(label_train).astype(np.float32)
y_val = lb.transform(label_val).astype(np.float32)
y_test = lb.transform(label_test).astype(np.float32)
# import ipdb; ipdb.set_trace()

# model1 = WaveletNeuralNetworkClassifier(
#     x_train.shape[1],
#     n_wavelets=64,
#     wavelet_range=[k for k in range(1, 5)],
#     wavelet_length=16,
#     conv_structure=[
#         (2, 4, 1, 2, 64, 'selu'),
#         (2, 4, 1, 2, 64, 'selu'),
#         (1, 4, 1, 4, -1, 'pooling'),
#         (2, 8, 1, 4, 128, 'selu'),
#         (4, 16, 2, 8, 128, 'selu'),
#         (2, 4, 2, 4, -1, 'pooling'),
#     ],
#     dense_structure=[
#         (1024, 'selu'),
#         (1024, 'selu'),
#     ],
#     output_dim=y_train.shape[1],
#     wavelet_dropout_prob=0.0,
#     conv_dropout_prob=0.0,
#     dense_dropout_prob=0.5,
# )
model_2 = WaveletNeuralNetworkClassifier(
    x_train.shape[1],
    n_wavelets=32,
    wavelet_range=[4*k for k in range(1, 25)],
    wavelet_length=-1,
    conv_structure=[
        (2, 4, 1, 2, 32, 'selu'),
        (2, 4, 1, 2, 32, 'selu'),
        (2, 4, 1, 2, 64, 'selu'),
        # (1, 2, 1, 2, -1, 'pooling'),
        (2, 4, 1, 2, 64, 'selu'),
        (2, 4, 1, 2, 64, 'selu'),
        (2, 8, 1, 2, 128, 'selu'),
        # (1, 2, 1, 2, -1, 'pooling'),
        (4, 16, 1, 2, 128, 'selu'),
        (4, 16, 1, 2, 256, 'selu'),
        (1, 2, 1, 2, -1, 'pooling'),
    ],
    dense_structure=[
        (2048, 'selu'),
        (2048, 'selu'),
    ],
    output_dim=y_train.shape[1],
    share_wavelet=False,
    wavelet_dropout_prob=0.0,
    conv_dropout_prob=0.0,
    dense_dropout_prob=0.1,
)

def main():
    # with open(GENERATED_RAW_DATA_PATH, 'rb') as f:
    # import ipdb; ipdb.set_trace()
    batch_gen = BatchGenerator(x_train, y_train, sample_weight_train, batch_size=32, augmented=True)
    clf = model_2
    clf.fit_generator(
        batch_gen,
        x_subtrain=x_train,
        y_subtrain=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=100,
        learning_rate=0.1,
        early_stopping_rounds=10,
        save_best=True,
        save_folder=MODEL_DIR,
    )
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
