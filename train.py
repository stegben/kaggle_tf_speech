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
    args = parser.parse_args()
    return args


def get_data():
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
    batch_gen = AugmentationBatchGenerator(
        x_train,
        y_train,
        sample_weight_train,
        batch_size=32,
        augmented=True,
    )

    fit_generator(
        sess=sess,
        train_batch_gen=batch_gen,
        max_batches=1000000,
        evaluate_set=(
            (3000, x_train, y_train, False),
            (200, x_val, y_val, True),
            (1000, x_test, y_test, False),
        ),
        learning_rate=0.01,
        learning_rate_decay_ratio=1/3,
        learning_rate_decay_rounds=10,
        early_stopping_rounds=50,
        save_folder=MODEL_DIR,
        save_best=True,
        model_name_prefix=args.model_id,
        saver=saver,
        **default_fit_params,
    )
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
