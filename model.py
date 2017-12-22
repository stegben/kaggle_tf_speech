import os.path as osp
import shutil
from tempfile import mkdtemp
from datetime import datetime
from typing import Iterable, Tuple
import logging
import pickle as pkl

import numpy as np
from sklearn.metrics import confusion_matrix
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
    SAMPLE_WEIGHT_PLACE = 'input_sample_weight_place'
    LR_PLACE = 'input_lr_placeholder'
    WAVELET_DROPOUT_PLACE = 'input_wavelet_dropout_placeholder_with_default_0'
    CONV_DROPOUT_PLACE = 'input_conv_dropout_placeholder_with_default_0'
    DENSE_DROPOUT_PLACE = 'input_dropout_placeholder_with_default_0'

    OP_INFERENCE = 'op_inference'
    OP_LOSS = 'op_loss'
    OP_TRAIN = 'op_train'

    def __init__(
            self,
            input_dim: int,
            n_wavelets: int,
            wavelet_length: int,
            wavelet_range: Iterable[int],
            conv_structure: Iterable[Tuple[int, int, int, int, int, str]],  # width, height, stride_width, stride_height, n_kernel, activation
            dense_structure: Iterable[Tuple[int, str]],  # neuron, activation
            output_dim: int,
            share_wavelet: bool = True,
            l2_regularize: float = 0.00001,
            wavelet_dropout_prob: float = 0.2,
            conv_dropout_prob: float = 0.2,
            dense_dropout_prob: float = 0.5,
            logger=LOGGER,
            seed_base=2017,
            name: str = 'clf',
        ):
        self.input_dim = input_dim
        self.n_wavelets = n_wavelets
        self.wavelet_length = wavelet_length
        self.wavelet_range = wavelet_range
        self.conv_structure = conv_structure
        self.dense_structure = dense_structure
        self.output_dim = output_dim

        self.share_wavelet = share_wavelet
        self.l2_regularize = l2_regularize
        self.wavelet_dropout_prob = wavelet_dropout_prob
        self.conv_dropout_prob = conv_dropout_prob
        self.dense_dropout_prob = dense_dropout_prob
        self.logger = logger
        self.seed_base = seed_base
        self.name = name

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
        # inputs
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
        sample_weight_place = tf.placeholder(
            dtype=tf.float32,
            shape=[None],
            name=self.SAMPLE_WEIGHT_PLACE,
        )
        lr_place = tf.placeholder(
            dtype=tf.float32,
            shape=(),  # means a scaler
            name=self.LR_PLACE,
        )
        wavelet_dropout_place = tf.placeholder_with_default(
            0.0,
            shape=(),
            name=self.WAVELET_DROPOUT_PLACE,
        )
        conv_dropout_place = tf.placeholder_with_default(
            0.0,
            shape=(),
            name=self.CONV_DROPOUT_PLACE,
        )
        dense_dropout_place = tf.placeholder_with_default(
            0.0,
            shape=(),
            name=self.DENSE_DROPOUT_PLACE,
        )

        # wavelet layers
        x_place_reshape = tf.expand_dims(x_place, axis=1)
        print(x_place_reshape.shape)
        if self.share_wavelet:
            wavelets = tf.get_variable(
                'wavelet_weights',
                shape=[self.wavelet_length, 1, self.n_wavelets],  # [wavelet_size, n_channel, n_wavelet]
                initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base - 1),
            )
        batch_size = x_place_reshape.shape[0]
        imfs = []
        for k in self.wavelet_range:
            if not self.share_wavelet:
                wavelets = tf.get_variable(
                    'wavelet_weights_{}'.format(k),
                    shape=[k, 1, self.n_wavelets],  # [wavelet_size, n_channel, n_wavelet]
                    initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base - 1),
                )
                imf = tf.nn.convolution(
                    input=x_place_reshape,
                    filter=wavelets,
                    padding='SAME',
                    strides=None,
                    dilation_rate=None,
                    name="wavelet_1d_conv_{}".format(k),
                    data_format='NCW'
                )
            else:
                imf = tf.nn.convolution(
                    input=x_place_reshape,
                    filter=wavelets,
                    padding='SAME',
                    strides=None,
                    dilation_rate=(k,),
                    name="wavelet_1d_conv_{}".format(k),
                    data_format='NCW'
                )
            pooled_imf = tf.layers.average_pooling1d(
                tf.transpose(imf, perm=[0, 2, 1]),
                pool_size=(4,),
                strides=(4,),
                data_format='channels_first',
                name="wavelet_1d_pool_{}".format(k),
            )
            imfs.append(pooled_imf)
        wavelet_out = tf.stack(imfs, axis=1)
        wavelet_out = tf.nn.tanh(wavelet_out)
        wavelet_out = tf.nn.dropout(wavelet_out, keep_prob=(1 - wavelet_dropout_place))
        print(imf.shape)
        print(pooled_imf.shape)
        print(wavelet_out.shape)

        # conv layers
        conv_out = wavelet_out
        n_input_channel = self.n_wavelets
        for n_layer, (w, h, sw, sh, n_kernel, activation) in enumerate(self.conv_structure):
            if activation == 'pooling':
                conv_out = tf.layers.max_pooling2d(
                    conv_out,
                    pool_size=(w, h),
                    strides=(sw, sh),
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )
                conv_out = tf.nn.dropout(conv_out, keep_prob=(1 - conv_dropout_place))
                print(conv_out.shape)
            else:
                kernel = tf.get_variable(
                    'kernel_weights_{}'.format(n_layer+1),
                    shape=[w, h, n_input_channel, n_kernel],
                    initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base + n_layer),
                )
                conv_out = tf.nn.convolution(
                    input=conv_out,
                    filter=kernel,
                    padding='SAME',
                    strides=(sw, sh),
                    dilation_rate=None,
                    name='conv1',
                    data_format='NHWC'
                )
                print(conv_out.shape)
                conv_out = ACTIVATIONS[activation](conv_out)
                n_input_channel = n_kernel

            print(conv_out.shape)

        # Dense Layer
        a = tf.layers.flatten(conv_out)
        print(a.shape)
        dense_input_dim = a.shape[1]
        l2_loss_dense = 0
        for n_layer, (n_neuron, activation) in enumerate(self.dense_structure):
            dense_output_dim = n_neuron
            weights = tf.get_variable(
                'weights_{}'.format(n_layer + 1),
                shape=[dense_input_dim, dense_output_dim],
                initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base + n_layer),
            )
            biases = tf.get_variable(
                'biases_{}'.format(n_layer + 1),
                shape=[dense_output_dim],
                initializer=tf.zeros_initializer(),
            )
            a = ACTIVATIONS[activation](a @ weights + biases)
            a = tf.nn.dropout(a, keep_prob=(1 - dense_dropout_place))
            l2_loss_dense += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            dense_input_dim = dense_output_dim

        # output layer
        weights_output = tf.get_variable(
            'weights_output',
            shape=[dense_output_dim, self.output_dim],
            initializer=tf.keras.initializers.lecun_uniform(seed=self.seed_base + 3),
        )
        biases_output = tf.get_variable(
            'biases_output',
            shape=[self.output_dim],
            initializer=tf.zeros_initializer(),
        )
        l2_loss_output = tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_output)
        output_before_softmax = a @ weights_output + biases_output
        output_ = tf.nn.softmax(output_before_softmax)
        output = tf.identity(output_, name=self.OP_INFERENCE)

        # get loss
        loss_ = tf.losses.softmax_cross_entropy(
            y_place,
            output_before_softmax,
            weights=sample_weight_place,
            label_smoothing=0.0,
        )
        loss = tf.reduce_mean(
            loss_,
            name=self.OP_LOSS,
        )
        loss = loss + self.l2_regularize * (l2_loss_dense + l2_loss_output)

        # training
        tf.train.AdadeltaOptimizer(lr_place).minimize(
            loss,
            name=self.OP_TRAIN,
        )

    def fit_generator(
            self,
            train_gen,
            x_subtrain: np.ndarray,
            y_subtrain: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            batch_size: int = 8,
            epochs: int = 100,
            learning_rate: float = 0.001,
            early_stopping_rounds: int = 10,
            save_folder: str = mkdtemp(),
            save_best: bool = True,
            save_best_after: int = 3,
        ) -> None:
        batch_gen = train_gen

        waiting_rounds = 0
        best_validation_loss = float('inf')

        for epoch in range(epochs):
            for x_batch, y_batch, weight_batch in tqdm(batch_gen()):
                self.fit_batch(x_batch, y_batch, weight_batch, learning_rate)
            current_subtrain_loss = self.evaluate(x_subtrain, y_subtrain)
            current_validation_loss = self.evaluate(x_val, y_val)

            subtrain_ans = y_subtrain.argmax(axis=1)
            subtrain_pred = self.predict(x_subtrain).argmax(axis=1)
            current_subtrain_accuracy = (subtrain_ans == subtrain_pred).mean()

            val_ans = y_val.argmax(axis=1)
            val_pred = self.predict(x_val).argmax(axis=1)
            current_validation_accuracy = (val_ans == val_pred).mean()
            print('Epoch {} subtrain loss: {}, subtrain accuracy: {}, validation loss: {}, validation accuracy: {}'.format(
            # self.logger.info('Epoch {} validation loss: {}, validation accuracy: {}'.format(
            # self.logger.info('Epoch {} train loss: {}, validation loss: {}, validation accuracy: {}'.format(
                epoch,
                current_subtrain_loss,
                current_subtrain_accuracy,
                current_validation_loss,
                current_validation_accuracy,
            ))
            print(confusion_matrix(subtrain_ans, subtrain_pred))
            print(confusion_matrix(val_ans, val_pred))

            if current_validation_loss < best_validation_loss:
                print('Improved!')
                waiting_rounds = 0
                # save best model
                if save_best and (epoch >= save_best_after):
                    print('save best')
                    model_name = '{}__epoch_{}_at_{}__tracc_{}__valacc_{}.mdl'.format(
                        self.name,
                        epoch,
                        datetime.now().strftime('%Y%m%d-%H%M%S'),
                        current_subtrain_accuracy,
                        current_validation_accuracy,
                    )
                    best_variable_path = osp.join(save_folder, model_name)
                    self.saver.save(self.sess, best_variable_path)
                best_validation_loss = current_validation_loss
            else:
                if waiting_rounds >= early_stopping_rounds:
                    self.logger.info('Early Stop')
                    print('Early Stop')
                    break
                waiting_rounds += 1
        if save_best:
            print('restore')
            self.saver.restore(self.sess, best_variable_path)
            # shutil.rmtree(temp_folder)

    def fit_batch(self, x_batch, y_batch, weight_batch, learning_rate):
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        y_place = self.graph.get_tensor_by_name(self.Y_PLACE + ':0')
        sample_weight_place = self.graph.get_tensor_by_name(self.SAMPLE_WEIGHT_PLACE + ':0')
        lr_place = self.graph.get_tensor_by_name(self.LR_PLACE + ':0')
        wavelet_dropout_place = self.graph.get_tensor_by_name(self.WAVELET_DROPOUT_PLACE + ':0')
        conv_dropout_place = self.graph.get_tensor_by_name(self.CONV_DROPOUT_PLACE + ':0')
        dense_dropout_place = self.graph.get_tensor_by_name(self.DENSE_DROPOUT_PLACE + ':0')
        loss_tensor = self.graph.get_tensor_by_name(self.OP_LOSS + ':0')
        train_op = self.graph.get_operation_by_name(self.OP_TRAIN)

        _, batch_loss = self.sess.run(
            [train_op, loss_tensor],
            feed_dict={
                x_place: x_batch,
                y_place: y_batch,
                sample_weight_place: weight_batch,
                lr_place: learning_rate,
                wavelet_dropout_place: self.wavelet_dropout_prob,
                conv_dropout_place: self.conv_dropout_prob,
                dense_dropout_place: self.dense_dropout_prob,
            },
        )

        self.logger.debug('batch training loss: {}'.format(batch_loss))
        return batch_loss

    def evaluate(self, x_val, y_val, batch_size=128):
        loss = []
        batch_gen = BatchGenerator(
            x=x_val,
            y=y_val,
            batch_size=batch_size,
            shuffle=False,
            augmented=False,
        )
        for x_batch, y_batch in tqdm(batch_gen()):
            loss.append(self.evaluate_batch(x_batch, y_batch))
        loss = np.mean(loss)
        return loss

    def evaluate_batch(self, x_batch, y_batch):
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        y_place = self.graph.get_tensor_by_name(self.Y_PLACE + ':0')
        sample_weight_place = self.graph.get_tensor_by_name(self.SAMPLE_WEIGHT_PLACE + ':0')
        loss_op = self.graph.get_operation_by_name(self.OP_LOSS)
        loss_tensor = self.graph.get_tensor_by_name(self.OP_LOSS + ':0')
        _, batch_loss = self.sess.run(
            [loss_op, loss_tensor],
            feed_dict={
                x_place: x_batch,
                y_place: y_batch,
                sample_weight_place: np.ones(x_batch.shape[0]),
            },
        )
        return batch_loss

    def predict(self, x_test, batch_size=128):
        batch_gen = BatchGenerator(
            x=x_test,
            y=np.empty_like(x_test),
            batch_size=batch_size,
            shuffle=False,
            augmented=False,
        )
        result = []

        for x_batch, _ in tqdm(batch_gen()):
            batch_result = self.predict_batch(x_batch)
            result.append(batch_result)
        return np.concatenate(result, axis=0)

    def predict_batch(self, x_batch):
        inference_op = self.graph.get_operation_by_name(self.OP_INFERENCE)
        inference_tensor = self.graph.get_tensor_by_name(
            self.OP_INFERENCE + ':0')
        x_place = self.graph.get_tensor_by_name(self.X_PLACE + ':0')
        sample_weight_place = self.graph.get_tensor_by_name(self.SAMPLE_WEIGHT_PLACE + ':0')
        _, batch_result = self.sess.run(
            [inference_op, inference_tensor],
            feed_dict={
                x_place: x_batch,
                sample_weight_place: np.ones(x_batch.shape[0]),

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
