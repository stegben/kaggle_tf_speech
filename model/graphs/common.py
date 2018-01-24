import tensorflow as tf
from ..names import (
    X_PLACE,
    Y_PLACE,
    SAMPLE_WEIGHT_PLACE,
    LR_PLACE,
    WAVELET_DROPOUT_PLACE,
    CONV_DROPOUT_PLACE,
    DENSE_DROPOUT_PLACE,
    IS_TRAINING_PLACE,
)


ACTIVATIONS = {
    'selu': tf.nn.selu,
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'sigmoid': tf.nn.sigmoid,
}


def get_input(input_dim, output_dim, divide_std=True):
    x_place = tf.placeholder(
        dtype=tf.float32,
        shape=[None, input_dim],
        name=X_PLACE,
    )
    if divide_std:
        x_place = x_place / 3128.12
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
    wavelet_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=WAVELET_DROPOUT_PLACE,
    )
    conv_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=CONV_DROPOUT_PLACE,
    )
    dense_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=DENSE_DROPOUT_PLACE,
    )
    is_training = tf.placeholder_with_default(
        False,
        shape=(),
        name=IS_TRAINING_PLACE,
    )
    return (
        x_place,
        y_place,
        sample_weight_place,
        lr_place,
        wavelet_dropout_place,
        conv_dropout_place,
        dense_dropout_place,
        is_training,
    )


def conv1d_layer(
        input_wave,
        is_training,
        kernel_width,
        n_kernels,
        activation,
        name,
        seed_base=2017,
    ):
    input_wave = tf.layers.batch_normalization(
        input_wave,
        axis=-1,
        training=is_training,
        fused=True,
    )
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


def conv2d_layer(
        input_wave,
        is_training,
        kernel_height,
        kernel_width,
        n_kernels,
        activation,
        name,
        seed_base=2017,
    ):
    input_wave = tf.layers.batch_normalization(
        input_wave,
        axis=1,
        training=is_training,
        fused=True,
    )
    kernel = tf.get_variable(
        name + 'kernel',
        shape=[kernel_height, kernel_width, input_wave.shape[1], n_kernels],
        initializer=tf.keras.initializers.lecun_uniform(seed=seed_base - 1),
    )
    conved = tf.nn.convolution(
        input=input_wave,
        filter=kernel,
        padding='SAME',
        strides=None,
        dilation_rate=None,
        name=name + '_conv_op',
        data_format='NCHW'
    )
    bias = tf.get_variable(
        name + 'bias',
        shape=(conved.shape[1],),
        initializer=tf.zeros_initializer(),
    )
    conved = tf.nn.bias_add(conved, bias, 'NCHW')
    activation = ACTIVATIONS[activation]
    return activation(conved)


def gated_conv1d_layer(
        input_wave,
        is_training,
        kernel_width,
        n_kernels,
        name,
        seed_base=2017,
    ):
    conved = conv1d_layer(
        input_wave,
        is_training,
        kernel_width,
        n_kernels,
        activation='tanh',
        name=name,
    )
    gated_conved = conv1d_layer(
        input_wave,
        is_training,
        kernel_width,
        n_kernels,
        activation='sigmoid',
        name=name+'_gate',
    )
    return tf.multiply(conved, gated_conved)


def dense_1d_block(
        input_wave,
        is_training,
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
                is_training,
                1,
                n_compressed_kernels,
                compress_layer_name,
            )
            conv_layer_name = name + str(idx) + '_conv'
            a = gated_conv1d_layer(
                a_compressed,
                is_training,
                window_length,
                n_kernels,
                conv_layer_name,
            )
            outputs.append(a)
        else:
            compress_layer_name = name + str(idx) + '_compressed'
            a_compressed = conv1d_layer(
                a_with_prev_outputs,
                is_training,
                1,
                n_compressed_kernels,
                activation=activation,
                name=compress_layer_name,
            )
            conv_layer_name = name + str(idx) + '_conv'
            a = conv1d_layer(
                a_compressed,
                is_training,
                window_length,
                n_kernels,
                activation=activation,
                name=conv_layer_name,
            )
            outputs.append(a)
    return a


def dense_2d_block(
        input_wave,
        is_training,
        n_layers,
        n_kernels,
        n_compressed_kernels,
        kernel_height,
        kernel_width,
        activation,
        with_compressed=True,
        concat_instead=True,
        name='some_dense_2d_net',
    ):
    a_input = input_wave
    for idx in range(n_layers):
        print(a_input.shape)
        if with_compressed:
            compress_layer_name = name + str(idx) + '_compressed'
            a_input_to_conv = conv2d_layer(
                a_input,
                is_training,
                1,
                1,
                n_compressed_kernels,
                activation=activation,
                name=compress_layer_name,
            )
        else:
            a_input_to_conv = a_input
        print(a_input_to_conv.shape)
        conv_layer_name = name + str(idx) + '_conv'
        a = conv2d_layer(
            a_input_to_conv,
            is_training,
            kernel_height,
            kernel_width,
            n_kernels,
            activation=activation,
            name=conv_layer_name,
        )
        if concat_instead:
            a_input = tf.concat([a, a_input], axis=1)
        else:
            # TODO: fix dimension not match error
            # a_input = a - a_input
            a_input = tf.concat([a, a_input], axis=1)
    return a
