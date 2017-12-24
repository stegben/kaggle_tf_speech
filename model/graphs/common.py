import tensorflow as tf
from ..names import (
    X_PLACE,
    Y_PLACE,
    SAMPLE_WEIGHT_PLACE,
    LR_PLACE,
)


ACTIVATIONS = {
    'selu': tf.nn.selu,
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'sigmoid': tf.nn.sigmoid,
}


def get_input(input_dim, output_dim):
    x_place = tf.placeholder(
        dtype=tf.float32,
        shape=[None, input_dim],
        name=X_PLACE,
    )
    y_place = tf.placeholder(
        dtype=tf.float32,
        shape=[None, output_dim],
        name=Y_PLACE,
    )
    sample_weight_place = tf.placeholder(
        dtype=tf.float32,
        shape=[None],
        name=SAMPLE_WEIGHT_PLACE,
    )
    lr_place = tf.placeholder(
        dtype=tf.float32,
        shape=(),  # means a scaler
        name=LR_PLACE,
    )
    return x_place, y_place, sample_weight_place, lr_place


def conv1d_layer(
        input_wave,
        kernel_width,
        n_kernels,
        activation,
        name,
        seed_base=2017,
    ):
    kernel = tf.get_variable(
        name + 'kernel',
        shape=[kernel_width, input_wave.shape[1], n_kernels],
        # [wavelet_size, n_channel, n_wavelet]
        initializer=tf.keras.initializers.lecun_uniform(seed=seed_base - 1),
    )
    conved = tf.nn.convolution(
        input=input_wave,
        filter=kernel,
        padding='SAME',
        strides=None,
        dilation_rate=None,
        name=name + '_conv_op',
        data_format='NCW'
    )
    activation = ACTIVATIONS[activation]
    return activation(conved)


def gated_conv1d_layer(
        input_wave,
        kernel_width,
        n_kernels,
        name,
        seed_base=2017,
    ):
    conved = conv1d_layer(
        input_wave,
        kernel_width,
        n_kernels,
        activation='tanh',
        name=name,
    )
    gated_conved = conv1d_layer(
        input_wave,
        kernel_width,
        n_kernels,
        activation='sigmoid',
        name=name+'_gate',
    )
    return tf.multiply(conved, gated_conved)


def dense_1d_block(
        input_wave,
        n_layers,
        n_kernels,
        n_compressed_kernels,
        window_length,
        activation,
        gated=False,
        name='some_dense_1d_net',
    ):
    a = input_wave
    outputs = []
    for idx in range(n_layers):
        a_with_prev_outputs = tf.concat(outputs + [a], axis=1)
        if gated:
            compress_layer_name = name + str(idx) + '_compressed'
            a_compressed = gated_conv1d_layer(
                a_with_prev_outputs,
                1,
                n_compressed_kernels,
                compress_layer_name,
            )
            conv_layer_name = name + str(idx) + '_conv'
            a = gated_conv1d_layer(
                a_compressed,
                window_length,
                n_kernels,
                conv_layer_name,
            )
            outputs.append(a)
        else:
            compress_layer_name = name + str(idx) + '_compressed'
            a_compressed = conv1d_layer(
                a_with_prev_outputs,
                1,
                n_compressed_kernels,
                activation=activation,
                name=compress_layer_name,
            )
            conv_layer_name = name + str(idx) + '_conv'
            a = conv1d_layer(
                a_compressed,
                window_length,
                n_kernels,
                activation=activation,
                name=conv_layer_name,
            )
            outputs.append(a)
    return a



