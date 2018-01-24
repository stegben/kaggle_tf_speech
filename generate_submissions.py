import os
import pickle as pkl
import argparse

import numpy as np
import pandas as pd

from model.utils import restore_session
from model.methods import predict
from utils import STD, AugmentationBatchGenerator
from constants import (
    GENERATED_RAW_DATA_PATH,
    LABEL_ENCODER_PATH,
    SUBMISSIONS_DIR,
)


def train_argparser():
    parser = argparse.ArgumentParser(
        description=('Generate'),
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
        '-bs',
        '--batch_size',
        type=int,
        help='batch size, default 128',
        default=128,
    )
    parser.add_argument(
        '-au',
        '--augment_times',
        type=int,
        help=' [None]',
        default=None,
    )
    args = parser.parse_args()
    return args


def get_data():
    with open(GENERATED_RAW_DATA_PATH, 'rb') as f:
        data = pkl.load(f)

    _, _, _, subs = data
    x_sub_fname, fname_sub = subs
    x_sub = np.memmap(x_sub_fname, dtype=np.int16, shape=(len(fname_sub), 16000), mode='r')
    return x_sub, fname_sub


def main():
    args = train_argparser()
    x_sub, fname_sub = get_data()
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        lb = pkl.load(f)

    augment_times = args.augment_times
    gen = AugmentationBatchGenerator(
        x_sub,
        np.ones((x_sub.shape[0], 1)),
        np.ones((x_sub.shape[0], 1)),
        batch_size=x_sub.shape[0],
        shuffle=False,
        augmented=True,
    )
    sess_paths = args.sess_path.split(',')
    pred_sub = np.zeros((x_sub.shape[0], 12), dtype=np.float32)
    model_names = []
    for sess_path in sess_paths:
        _, sess = restore_session(sess_path)
        model_name = args.sess_path.split('/')[-1]
        model_names.append(model_name)
        if augment_times is None:
            pred_sub += predict(sess, x_sub, batch_size=args.batch_size)
        else:
            for idx, (x_sub_batch, _, __) in enumerate(gen()):
                pred_sub += predict(sess, x_sub_batch, batch_size=args.batch_size)
                if (idx + 1) >= augment_times:
                    break

    pred_ans = pred_sub.argmax(axis=1)
    pred_label = [lb.classes_[ans] for ans in pred_ans]
    sub = pd.DataFrame({
        'fname': fname_sub,
        'label': pred_label,
    })
    sub_fname = '___'.join(model_names) + '.csv'
    if augment_times is not None:
        sub_fname = 'aug_{}_'.format(augment_times) + sub_fname
    sub.to_csv(sub_fname, index=False)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
