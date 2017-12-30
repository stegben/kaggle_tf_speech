import os
import argparse
from collections import Counter
import pickle as pkl

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from model import ALL_MODELS
from model.methods import fit_generator
from model.utils import restore_session
from utils import AugmentationBatchGenerator
from constants import (
    GENERATED_RAW_DATA_PATH,
    MODEL_DIR,
    LABEL_ENCODER_PATH,
)


def train_argparser():
    parser = argparse.ArgumentParser(
        description=('Train Speech Recorgnizer'),
    )
    parser.add_argument(
        '-s',
        '--sess_path',
        type=str,
        help='path of previously trained session.'
        'If you are going to train a new one, do not specify this',
        default=None,
    )
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        help='The model_id.',
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help='learning rate',
        default=0.001,
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        help='batch size, default 32',
        default=32,
    )
    parser.add_argument(
        '-esr',
        '--early_stopping_rounds',
        type=int,
        help='Early Stopping Rounds, default 100',
        default=100,
    )
    parser.add_argument(
        '-lrdr',
        '--learning_rate_decay_rounds',
        type=int,
        help='Learning rate decay Rounds, default 30',
        default=30,
    )
    parser.add_argument(
        '-trbr',
        '--training_set_batch_rounds',
        type=int,
        help='Training set will be evaluate every n rounds. n default 3000',
        default=3000,
    )
    parser.add_argument(
        '-valbr',
        '--validation_set_batch_rounds',
        type=int,
        help='Validation set will be evaluate every n rounds. n default 200',
        default=200,
    )
    parser.add_argument(
        '-tebr',
        '--testing_set_batch_rounds',
        type=int,
        help='Testing set will be evaluate every n rounds. n default 1000',
        default=1000,
    )
    parser.add_argument(
        '-vat',
        '--val_augment_times',
        type=int,
        help='validation set augment times [4]',
        default=4,
    )
    args = parser.parse_args()
    return args


def get_data():
    with open(GENERATED_RAW_DATA_PATH, 'rb') as f:
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

    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            lb = pkl.load(f)
        y_train = lb.transform(label_train).astype(np.float32)
    else:
        lb = LabelBinarizer()
        y_train = lb.fit_transform(label_train).astype(np.float32)
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pkl.dump(lb, f)
    y_val = lb.transform(label_val).astype(np.float32)
    y_test = lb.transform(label_test).astype(np.float32)

    return x_train, y_train, sample_weight_train, x_val, y_val, x_test, y_test,


def main():
    args = train_argparser()

    if args.sess_path is None:
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)

        build_graph, default_fit_params = ALL_MODELS[args.model_id]

        with sess.graph.as_default():
            build_graph(
                16000,
                12,
            )
        saver = tf.train.Saver(
            var_list=graph.get_collection('variables'),
            max_to_keep=None,
        )
        sess.run(tf.variables_initializer(
            var_list=graph.get_collection('variables'),
        ))
    else:
        saver, sess = restore_session(args.sess_path)
        graph = sess.graph
        default_fit_params = {}

    x_train, y_train, sample_weight_train, x_val, y_val, x_test, y_test = get_data()
    # augment x_val
    val_gen = AugmentationBatchGenerator(
        x_val,
        y_val,
        np.ones((x_val.shape[0], 1)),
        batch_size=y_val.shape[0],
        shuffle=False,
        augmented=True
    )
    x_val_augmented = []
    y_val_augmented = []
    for idx, (x_batch, y_batch, _) in enumerate(val_gen()):
        if idx > args.val_augment_times:
            break
        x_val_augmented.append(x_batch)
        y_val_augmented.append(y_batch)
    x_val_augmented = np.concatenate(x_val_augmented, axis=0)
    y_val_augmented = np.concatenate(y_val_augmented, axis=0)

    batch_gen = AugmentationBatchGenerator(
        x_train,
        y_train,
        sample_weight_train,
        batch_size=args.batch_size,
        augmented=True,
    )

    fit_generator(
        sess=sess,
        train_batch_gen=batch_gen,
        max_batches=1000000,
        evaluate_set=(
            (args.training_set_batch_rounds, x_train, y_train, False),
            (args.validation_set_batch_rounds, x_val_augmented, y_val_augmented, True),
            (args.validation_set_batch_rounds, x_val, y_val, False),
            (args.testing_set_batch_rounds, x_test, y_test, False),
        ),
        learning_rate=args.learning_rate,
        learning_rate_decay_ratio=1/2,
        learning_rate_decay_rounds=args.learning_rate_decay_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        save_folder=MODEL_DIR,
        save_best=True,
        model_name_prefix=args.model_id,
        saver=saver,
        **default_fit_params,
    )
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
