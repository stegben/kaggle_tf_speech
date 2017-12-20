import os.path as osp
import shutil
from tempfile import mkdtemp
from typing import Iterable, Tuple
import logging
import pickle as pkl

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import BatchGenerator


LOGGER = logging.getLogger(__name__)

ACTIVATIONS = {
    'selu': tf.nn.selu,
    'relu': tf.nn.relu,
}


class WaveletNeuralNetworkClassifier:

    X_PLACE = 'input_x_placeholder'
    Y_PLACE = 'input_y_placeholder'
    LR_PLACE = 'input_lr_placeholder'
    DROPOUT_PLACE = 'input_dropout_placeholder_with_default_0'

    OP_INFERENCE = 'op_inference'
    OP_LOSS = 'op_loss'
    OP_TRAIN = 'op_train'

    def __init__(
            self,
            input_dim: int,
            n_wavelets: int,
            wavelet_length: int,
            output_dim: int,
            l2_regularize: float = 0.001,
            dropout_prob: float = 0.2,
            logger=LOGGER,
            seed_base=2017,
        ):
        self.input_dim = input_dim
        self.n_wavelets = n_wavelets
        self.wavelet_length = wavelet_length
        self.output_dim = output_dim
        self.l2_regularize = l2_regularize
        self.dropout_prob = dropout_prob
        self.logger = logger
        self.seed_base = seed_base

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/device:GPU:0'):
                self._build_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(tf.variables_initializer(
            var_list=self.graph.get_collection('variables'),
        ))
        self.saver = tf.train.Saver(
            var_list=self.graph.get_collection('variables'),
            max_to_keep=None,
        )

    def _build_graph(self):
        x_place = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_dim],
            name=self.X_PLACE,
        )
        y_place = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.output_dim],
            name=self.Y_PLACE,
        )
        lr_place = tf.placeholder(
            dtype=tf.float32,
            shape=(),  # means a scaler
            name=self.LR_PLACE,
        )
        dropout_place = tf.placeholder_with_default(
            0.0,
            shape=(),
            name=self.DROPOUT_PLACE,
        )

        x_place_reshape = tf.expand_dims(x_place, axis=1)

        depth = self.n_wavelets

        wavelets = tf.get_variable(
            'wavelet_weights',
            shape=[self.wavelet_length, 1, depth],  # [wavelet_size, n_channel, n_wavelet]
            initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base),
        )

        imfs = []
        for k in range(1, 33):
            imf = tf.nn.convolution(
                input=x_place_reshape,
                filter=wavelets*k,
                padding='SAME',
                strides=None,
                dilation_rate=(k,),
                name=None,
                data_format='NCW'
            )
            pooled_imf = tf.layers.max_pooling1d(
                tf.transpose(imf, perm=[0, 2, 1]),
                pool_size=(4,),
                strides=(4,),
                data_format='channels_first',
            )
            print(imf.shape)
            print(pooled_imf.shape)
            imfs.append(pooled_imf)

        new_tensor = tf.stack(imfs, axis=1)

        kernel_1 = tf.get_variable(
            'kernel_weights',
            shape=[3, 3, depth, 32],  # [wavelet_size, n_channel, n_wavelet]
            initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base),
        )
        out1 = tf.nn.convolution(
            input=new_tensor,
            filter=kernel_1,
            padding='SAME',
            strides=(2, 2),
            dilation_rate=None,
            name=None,
            data_format='NHWC'
        )
        print(out1.shape)
        out1 = tf.nn.selu(out1)
        out1 = tf.layers.max_pooling2d(
            out1,
            pool_size=(3, 16),
            strides=(3, 16),
            padding='valid',
            data_format='channels_last',
            name=None
        )
        print(out1.shape)

        a = tf.layers.flatten(out1)
        print(a.shape)
        # import ipdb; ipdb.set_trace()
        # output layer
        weights_output = tf.get_variable(
            'weights_output',
            shape=[a.shape[1], self.output_dim],
            initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base + 3),
        )
        biases_output = tf.get_variable(
            'biases_output',
            shape=[self.output_dim],
            initializer=tf.zeros_initializer(),
        )
        # l2_loss_output = tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_output)
        output_ = tf.nn.softmax(a @ weights_output + biases_output)
        output = tf.identity(output_, name=self.OP_INFERENCE)

        # get loss
        loss_ = tf.losses.softmax_cross_entropy(
            y_place,
            output,
            label_smoothing=0.0,
        )
        loss = tf.reduce_mean(
            loss_,
            name=self.OP_LOSS,
        )
        # loss = loss + self.l2_regularize * (l2_loss_hidden + l2_loss_output)

        # training
        tf.train.AdadeltaOptimizer(lr_place).minimize(
            loss,
            name=self.OP_TRAIN,
        )

    def fit_generator(
            self,
            train_gen,
            x_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 8,
            epochs: int = 100,
            learning_rate: float = 0.001,
            early_stopping_rounds: int = 10,
            save_best: bool = True,
        ) -> None:
        batch_gen = train_gen

        waiting_rounds = 0
        best_validation_loss = float('inf')
        if save_best:
            temp_folder = mkdtemp()
            temp_best_variable_path = osp.join(temp_folder, 'best_model')
        for epoch in range(epochs):
            for x_batch, y_batch in tqdm(batch_gen()):
                self.fit_batch(x_batch, y_batch, learning_rate)
            current_validation_loss = self.evaluate(x_val, y_val)
            current_validation_accuracy = (y_val.argmax(axis=1) == self.predict(x_val).argmax(axis=1)).mean()
            # current_train_loss = self.evaluate(x_tr, y_tr)
            print('Epoch {} validation loss: {}, validation accuracy: {}'.format(
            # self.logger.info('Epoch {} validation loss: {}, validation accuracy: {}'.format(
            # self.logger.info('Epoch {} train loss: {}, validation loss: {}, validation accuracy: {}'.format(
                epoch,
                # current_train_loss,
                current_validation_loss,
                current_validation_accuracy,
            ))

            if current_validation_loss < best_validation_loss:
                waiting_rounds = 0
                # save best model
                if save_best:
                    self.saver.save(self.sess, temp_best_variable_path)
                best_validation_loss = current_validation_loss
            else:
                if waiting_rounds >= early_stopping_rounds:
                    self.logger.info('Early Stop')
                    break
                waiting_rounds += 1
        if save_best:
            self.saver.restore(self.sess, temp_best_variable_path)
            shutil.rmtree(temp_folder)

    def fit_batch(self, x_batch, y_batch, learning_rate):
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        y_place = self.graph.get_tensor_by_name(self.Y_PLACE + ':0')
        lr_place = self.graph.get_tensor_by_name(self.LR_PLACE + ':0')
        dropout_place = self.graph.get_tensor_by_name(self.DROPOUT_PLACE + ':0')
        loss_tensor = self.graph.get_tensor_by_name(self.OP_LOSS + ':0')
        train_op = self.graph.get_operation_by_name(self.OP_TRAIN)

        run_metadata = tf.RunMetadata()
        _, batch_loss = self.sess.run(
            [train_op, loss_tensor],
            feed_dict={
                x_place: x_batch,
                y_place: y_batch,
                lr_place: learning_rate,
                dropout_place: self.dropout_prob,
            },
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata,
        )
        from tensorflow.python.client import timeline
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        import ipdb; ipdb.set_trace()

        self.logger.debug('batch training loss: {}'.format(batch_loss))
        return batch_loss

    def evaluate(self, x_val, y_val, batch_size=32):
        loss = []
        batch_gen = BatchGenerator(
            x=x_val,
            y=y_val,
            batch_size=batch_size,
        )
        for x_batch, y_batch in batch_gen():
            loss.append(self.evaluate_batch(x_batch, y_batch))
        loss = np.mean(loss)
        return loss

    def evaluate_batch(self, x_batch, y_batch):
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        y_place = self.graph.get_tensor_by_name(self.Y_PLACE + ':0')
        loss_op = self.graph.get_operation_by_name(self.OP_LOSS)
        loss_tensor = self.graph.get_tensor_by_name(self.OP_LOSS + ':0')
        _, batch_loss = self.sess.run(
            [loss_op, loss_tensor],
            feed_dict={
                x_place: x_batch,
                y_place: y_batch,
            },
        )
        return batch_loss

    def predict(self, x_test, batch_size=1):
        batch_gen = BatchGenerator(
            x=x_test,
            y=np.empty_like(x_test),
            batch_size=batch_size,
            shuffle=False,
        )
        result = []

        for x_batch, _ in batch_gen():
            batch_result = self.predict_batch(x_batch)
            result.append(batch_result)
        return np.concatenate(result, axis=0)

    def predict_batch(self, x_batch):
        inference_op = self.graph.get_operation_by_name(self.OP_INFERENCE)
        inference_tensor = self.graph.get_tensor_by_name(
            self.OP_INFERENCE + ':0')
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        _, batch_result = self.sess.run(
            [inference_op, inference_tensor],
            feed_dict={
                x_place: x_batch,
            },
        )
        return batch_result

    def recover_from_cache(self) -> None:
        """Things to do when recover from cache.

        e.g. set Keras global session. (But plz, don't use keras)
        """
        pass

    @staticmethod
    def gen_path(path):
        hyper_param_path = path + '-param.pkl'
        variable_path = path + '-variable.model'
        return hyper_param_path, variable_path

    def save(self, path: str) -> None:
        hyper_param_path, variable_path = self.gen_path(path)
        with open(hyper_param_path, 'wb') as fw:
            params = {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_structure': self.hidden_structure,
            }
            pkl.dump(params, fw)
        self.saver.save(self.sess, variable_path)

    @classmethod
    def load(cls, path: str):
        """Load trained parameters.

        Use this method when load model to the cache.

        >>> clf = XXX.load('path/to/model')
        >>> y = clf.predict(x)
        """
        hyper_param_path, variable_path = cls.gen_path(path)
        with open(hyper_param_path, 'rb') as f:
            params = pkl.load(f)
        model = cls(**params)
        model.saver.restore(model.sess, variable_path)
        return model

    def __del__(self):
        """GC, session recycle, etc."""
        self.sess.close()
        del self.__dict__
