import os.path as osp
from datetime import datetime
from config import (
    DATA_PATH,
    GENERATED_DATA_PATH,
    SUBMISSIONS_DIR,
    MODEL_DIR,
)

TRAIN_PATH = osp.join(DATA_PATH, 'train')
TEST_PATH = osp.join(DATA_PATH, 'test')

TRAIN_AUDIO_PATH = osp.join(TRAIN_PATH, 'audio')
TRAIN_AUDIO_PATH_PATTERN = osp.join(TRAIN_AUDIO_PATH, '*/*.wav')
TEST_AUDIO_PATH = osp.join(TEST_PATH, 'audio')
TEST_AUDIO_PATH_PATTERN = osp.join(TEST_AUDIO_PATH, '*.wav')

TRAIN_VAL_LIST_PATH = osp.join(TRAIN_PATH, 'validation_list.txt')
TRAIN_TEST_LIST_PATH = osp.join(TRAIN_PATH, 'testing_list.txt')

with open(TRAIN_VAL_LIST_PATH, 'r') as f:
    TRAIN_VAL_FNAME = [fname.rstrip() for fname in f.readlines()]

with open(TRAIN_TEST_LIST_PATH, 'r') as f:
    TRAIN_TEST_FNAME = [fname.rstrip() for fname in f.readlines()]

TRAIN_VAL_PATH = []
for fname in TRAIN_VAL_FNAME:
    TRAIN_VAL_PATH.append(osp.join(TRAIN_AUDIO_PATH, fname))
TRAIN_TEST_PATH = []
for fname in TRAIN_TEST_FNAME:
    TRAIN_TEST_PATH.append(osp.join(TRAIN_AUDIO_PATH, fname))

TARGET_LABELS = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
]

SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'

LABELS = TARGET_LABELS + [SILENCE_LABEL] + [UNKNOWN_LABEL]

N_SAMPLE = 16000
SAMPLE_RATE = 8000

SILENCE_SLIDING_LENGTH = 1000

raw_data_fname = "RAW_DATA_{}_.pkl".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
GENERATED_RAW_DATA_PATH = osp.join(
    GENERATED_DATA_PATH,
    raw_data_fname,
)
