import numpy as np
import tensorflow as tf


class BatchGenerator(object):

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray = None,
            batch_size: int = 32,
            shuffle: bool = True,
            seed: int = 2017,
        ):
        self.x = x
        self.y = y
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.data_size = x.shape[0]
        self.shuffle = shuffle
        if shuffle:
            np.random.seed(seed)

    def __len__(self):
        return self.data_size // self.batch_size + 1

    def __call__(self):
        if self.shuffle:
            data_index = np.random.permutation(self.data_size)
        else:
            data_index = np.arange(self.data_size)

        for start_idx in range(0, self.data_size, self.batch_size):
            end_idx = start_idx + self.batch_size
            target_idxs = data_index[start_idx: end_idx]
            if self.sample_weight is None:
                yield (
                    self.x[target_idxs],
                    self.y[target_idxs],
                )
            else:
                yield (
                    self.x[target_idxs],
                    self.y[target_idxs],
                    self.sample_weight[target_idxs],
                )


def restore_session(path):
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)

    with graph.as_default():
        saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
        saver.restore(sess, path)

    return saver, sess


def save_session(sess, path, saver=None):
    with sess.graph.as_default():
        if saver is None:
            saver = tf.train.Saver(
                var_list=sess.graph.get_collection('variables'),
                max_to_keep=None,
            )
        saver.save(
            sess,
            save_path=path,
            global_step=None,
            latest_filename=None,
            meta_graph_suffix='meta',
            write_meta_graph=True,
            write_state=True
        )
